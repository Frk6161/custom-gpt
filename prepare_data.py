# prepare_data.py
with open('input.txt', 'r') as f:
    text = f.read()
split_idx = int(0.9 * len(text))
train_text = text[:split_idx]
val_text = text[split_idx:]

with open('train.txt', 'w') as f:
    f.write(train_text)
with open('val.txt', 'w') as f:
    f.write(val_text)
    
print("Train/Val Splits erstellt: train.txt, val.txt")