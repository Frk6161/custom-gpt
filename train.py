# train.py
import math
import time
import torch
from model import GPT
from config import Config, ModelConfig, TrainingConfig, OptimizerConfig
import inspect
import os
import wandb


def save_checkpoint(model, optimizer, step, loss, cfg, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "step":           step,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss":           loss,
        "model_config":   cfg.model,
        "train_config":   cfg.training,
    }
    filepath = os.path.join(path, f"ckpt_step{step:06d}.pt")
    torch.save(ckpt, filepath)
    print(f"Checkpoint gespeichert: {filepath}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Checkpoint geladen: step {ckpt['step']}, loss {ckpt['loss']:.4f}")
    return ckpt["step"] + 1  # nächster Step



def calculate_mfu(model, batch_size, seq_len, grad_accum_steps, dt_sec):
    """
    Grobe Schätzung der Model FLOPs Utilization (MFU).
    """
    # Anzahl der Parameter des Modells
    total_params = sum(p.numel() for p in model.parameters())
    
    # Anzahl Tokens, die in einem Schritt verarbeitet wurden
    tokens = batch_size * seq_len * grad_accum_steps
    
    # Ungefähre FLOPs pro Token 
    flops_per_token = 6 * total_params
    total_flops = flops_per_token * tokens
    
    # Theoretische maximale FLOPs deiner GPU pro Sekunde
    gpu_tflops = 82.6  
    max_flops = gpu_tflops * 1e12 * dt_sec
    
    mfu = total_flops / max_flops if max_flops > 0 else 0.0
    return min(mfu, 1.0)  # Begrenzung auf max. 1.0


@torch.no_grad()
def validate(model, val_loader, device, num_batches=5):
    model.eval()
    total_loss = 0.0
    for i in range(num_batches):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(scaler is not None)):
            _, loss, _ = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


#  1. Konfiguration laden 
cfg = Config.from_yaml("configs/")   # Pfad zum Ordner mit den YAMLs
model_cfg: ModelConfig = cfg.model
train_cfg: TrainingConfig = cfg.training
opt_cfg: OptimizerConfig = cfg.optimizer

wandb.init(
    project="custom-gpt",
    config={
        "n_layer":      model_cfg.n_embd,
        "n_head":       model_cfg.n_head,
        "n_embd":       model_cfg.n_embd,
        "batch_size":   train_cfg.micro_batch_size,
        "max_steps":    train_cfg.max_steps,
        "lr":           train_cfg.max_lr,
        "head_config":  str(model_cfg.head_config),
    }
)

#  2. Modell erstellen 
model = GPT(model_cfg)   # <-- übergib nur ModelConfig
if train_cfg.compile_model:   # if für debugging, vllt will man aus machen damit man schnell debuggen kann (compile dauert ja lange)
    model = torch.compile(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Modell auf {device} geladen.")

#  3. DataLoader 
from dataloader import DataLoaderLite
train_loader = DataLoaderLite(B=train_cfg.micro_batch_size, T=train_cfg.sequence_length)

#  4. Optimizer (eigenständig, nicht mehr Teil des Modells)
optimizer = model.configure_optimizers(
    weight_decay=train_cfg.weight_decay,
    learning_rate=opt_cfg.learning_rate,
    device=device,
    opt_cfg=opt_cfg
)
#  5. Lernratenplan 
max_lr = train_cfg.max_lr
min_lr = max_lr * train_cfg.min_lr_factor
warmup_steps = train_cfg.warmup_steps
max_steps = train_cfg.max_steps

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#  6. Training (mit AMP) 
scaler = torch.amp.GradScaler('cuda') if train_cfg.dtype == "float16" else None
dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
amp_dtype = dtype_map.get(train_cfg.dtype, torch.float16)

train_loader = DataLoaderLite(B=train_cfg.micro_batch_size, T=train_cfg.sequence_length, file_path='input.txt')
val_loader = DataLoaderLite(B=train_cfg.micro_batch_size, T=train_cfg.sequence_length, file_path='val.txt')  

start_step = 0
resume_path = train_cfg.resume_checkpoint  # None oder Pfad zum .pt-File
if resume_path and os.path.exists(resume_path):
    start_step = load_checkpoint(model, optimizer, resume_path)
    print(f"Training fortgesetzt ab Step {start_step}")

for step in range(start_step, max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(train_cfg.grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(scaler is not None)):
            logits, loss, _ = model(x, y)
        loss = loss / train_cfg.grad_accum_steps
        loss_accum += loss.detach()
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if scaler:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_processed = train_loader.B * train_loader.T * train_cfg.grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    # MFU berechnen
    mfu = calculate_mfu(model, train_loader.B, train_loader.T, train_cfg.grad_accum_steps, dt)
    print(f"step {step}, loss: {loss_accum.item():.4f}, lr {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.0f}, MFU: {mfu*100:.1f}%")

    wandb.log({
        "train/loss":       loss_accum.item(),
        "train/lr":         lr,
        "train/grad_norm":  norm,
        "perf/tokens_per_sec": tokens_per_sec,
        "perf/mfu":         mfu,
    }, step=step)
    if step % train_cfg.val_every == 0 and step > 0:
        val_loss = validate(model, val_loader, device, num_batches=train_cfg.val_samples)
        wandb.log({"val/loss": val_loss}, step=step)
        print(f"step {step:4d} | validation loss {val_loss:.4f}")



if step % train_cfg.val_every == 0 and step > 0:
    val_loss = validate(model, val_loader, device, num_batches=train_cfg.val_samples)
    print(f"step {step:4d} | validation loss {val_loss:.4f}")
    save_checkpoint(model, optimizer, step, val_loss, cfg)
print("Training abgeschlossen.")
wandb.finish()