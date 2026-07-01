import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
#from config import GPTConfig
from config import ModelConfig
from attention import CausalSelfAttention

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.swiglu:
            hidden_dim = int(4 * config.n_embd * 2 / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
            self.w1     = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.w2     = nn.Linear(hidden_dim, config.n_embd, bias=False)
            self.act    = nn.SiLU()
        else:
            self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu   = nn.GELU(approximate='tanh')
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        if hasattr(self, 'w1'):  # SwiGLU
            return self.w2(self.act(self.w_gate(x)) * self.w1(x))
        else:                    # Standard GELU
            return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config: ModelConfig, head_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, head_config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)   # MLP liest config.swiglu intern

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if not config.rope else nn.Identity(),
            h = nn.ModuleList([Block(config, config.head_config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        if self.config.rope:
            x = tok_emb  # keine Positionsembeddings nötig, RoPE macht das in Attention
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_head_config(self, head_config):
        self.config.head_config = head_config
        for block in self.transformer.h:
            old_attn = block.attn
            device = old_attn.c_attn.weight.device  # Gerät merken (z.B. cuda:0)
            new_attn = CausalSelfAttention(self.config, head_config)
            # Komplettes state_dict kopieren (c_attn, c_proj, bias)
            new_attn.load_state_dict(old_attn.state_dict())
            new_attn.to(device)  # auf das richtige Gerät schieben
            block.attn = new_attn


    def configure_optimizers(self, weight_decay, learning_rate, device, opt_cfg=None):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if opt_cfg is not None:
            use_fused = use_fused and opt_cfg.fused
        betas = opt_cfg.betas if opt_cfg else (0.9, 0.95)
        eps   = opt_cfg.eps   if opt_cfg else 1e-8
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused
        )
        return optimizer


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, \
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, \
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
