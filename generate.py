import torch
import torch.nn.functional as F
import tiktoken
from model import GPT
from config import Config
import os

def generate(model, device, prompt="Hello", max_new_tokens=100, num_sequences=1, temperature=1.0, top_k=50):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens

    kv_cache = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Beim ersten Step: gesamten Prompt verarbeiten
            # Danach: nur noch das letzte Token
            x_input = x if kv_cache is None else x[:, -1:]

            logits, _, kv_cache = model(x_input, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), top_k)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            x = torch.cat([x, next_token], dim=1)

    print("\nGenerierte Texte:")
    for i in range(num_sequences):
        decoded = enc.decode(x[i].tolist())
        print(f"\n[{i+1}] {decoded}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.from_yaml("configs/")
    model = GPT(cfg.model)
    
    # Neuesten Checkpoint laden
    ckpt_dir = "checkpoints"
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not ckpts:
        raise FileNotFoundError("Kein Checkpoint gefunden in checkpoints/")
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Checkpoint geladen: {ckpt_path}")

    model.to(device)
    model.eval()
    generate(model, device, prompt="Once upon a time", max_new_tokens=50, num_sequences=2)