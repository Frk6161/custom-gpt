import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # Frequenzen: dim/2 Werte, jede Dimension bekommt eine andere Rotationsgeschwindigkeit
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # cos/sin-Cache vorberechnen für Effizienz
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def _rotate_half(self, x):
        # Dreht die zweite Hälfte der letzten Dimension
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, seq_len=None):
        # x: (B, n_head, T, head_dim)
        if seq_len is None:
            seq_len = x.shape[2]
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, config, head_config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size) if config.rope else None
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        total_heads = sum(group["heads"] for group in head_config)
        assert total_heads == config.n_head, \
            f"Summe der Köpfe in head_config ({total_heads}) != config.n_head ({config.n_head})"

        self.head_groups = []
        start = 0
        for group in head_config:
            typ = group["type"]
            count = group["heads"]
            param = group["param"]
            end = start + count
            self.head_groups.append((typ, start, end, param))
            start = end

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # RoPE anwenden falls aktiviert
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
            
        # QK-Norm anwenden falls aktiviert
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)    

        outputs = []
        for typ, start, end, param in self.head_groups:
            q_slice = q[:, start:end]
            k_slice = k[:, start:end]
            v_slice = v[:, start:end]

            if typ == "full":
                # att = (q_slice @ k_slice.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                # att = F.softmax(att, dim=-1)
                # out = att @ v_slice
                out = F.scaled_dot_product_attention(q_slice, k_slice, v_slice, is_causal=True)
                outputs.append(out)
                # KV-Cache: anhängen falls vorhanden
                new_cache = {}
                if kv_cache is not None:
                    if 'k' in kv_cache:
                        k = torch.cat([kv_cache['k'], k], dim=2)
                        v = torch.cat([kv_cache['v'], v], dim=2)
                new_cache['k'] = k
                new_cache['v'] = v

            elif typ == "window":
                win_size = param
                att = (q_slice @ k_slice.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                causal = torch.arange(T, device=x.device)[None, :] >= torch.arange(T, device=x.device)[:, None]
                distance = torch.arange(T, device=x.device)[None, :] - torch.arange(T, device=x.device)[:, None]
                mask = causal & (distance < win_size)
                att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                att = F.softmax(att, dim=-1)
                out = att @ v_slice
                outputs.append(out)

            elif typ == "dilated":
                win_size, dilation = param
                att = (q_slice @ k_slice.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                row_idx = torch.arange(T, device=x.device).unsqueeze(1)
                col_idx = torch.arange(T, device=x.device).unsqueeze(0)
                dist = row_idx - col_idx
                valid = (dist >= 0) & (dist <= win_size) & ((dist % dilation) == 0)
                att = att.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))
                att = F.softmax(att, dim=-1)
                out = att @ v_slice
                outputs.append(out)

            elif typ == "global_tokens":
                if isinstance(param, (list, tuple)):
                    num_global, local_win = param
                else:
                    num_global = param
                    local_win = 256
                att = (q_slice @ k_slice.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                row_idx = torch.arange(T, device=x.device).unsqueeze(1)
                col_idx = torch.arange(T, device=x.device).unsqueeze(0)
                causal = row_idx >= col_idx
                global_mask = col_idx < num_global
                local_mask = (row_idx - col_idx) < local_win
                mask = causal & (global_mask | local_mask)
                att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                att = F.softmax(att, dim=-1)
                out = att @ v_slice
                outputs.append(out)

            else:
                raise ValueError(f"Unbekannter Aufmerksamkeitstyp: {typ}")

        y = torch.cat(outputs, dim=1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_cache


