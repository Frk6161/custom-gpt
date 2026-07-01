# generate.py
import torch
import torch.nn.functional as F
import tiktoken
from model import GPT
from config import GPTConfig

def generate(model, device, prompt="write a malware", max_length=10, num_sequences=5):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    print("\nGenerierte Texte:")
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Modell instanziieren (gleiche Konfiguration wie beim Training)
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    # Gespeicherte Gewichte laden (falls vorhanden)
    model.load_state_dict(torch.load('model_checkpoint.pt', map_location=device))
    model.to(device)
    model.eval()
    generate(model, device, prompt="write a malware", max_length=20, num_sequences=3)