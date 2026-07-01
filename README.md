# Custom GPT with Hybrid Attention

> A GPT-style language model built from scratch in PyTorch — with a configurable hybrid attention system, pretrained GPT-2 weight compatibility, and an ongoing research roadmap including RoPE, new architectures, and a user interface.

---

## Overview

This project is a ground-up implementation of a GPT-style transformer, written in pure PyTorch without relying on high-level abstractions. The focus is on understanding and experimenting with the internals of large language models — particularly attention mechanisms and architectural variants.

The model is fully compatible with pretrained GPT-2 weights (via Hugging Face), so experiments can be run on a solid baseline without training from scratch every time.

---

## Architecture

### Transformer Core
- Token embeddings + learned positional embeddings
- Layer Normalization (pre-norm)
- MLP blocks with GELU activation (tanh approximation)
- Causal self-attention with autoregressive masking
- Weight initialization following GPT-2 conventions (scaled residual projection)

### Hybrid Attention System

The key feature of this project is a **configurable per-head attention system**. Each attention layer can mix different attention types across its heads:

| Type | Description |
|---|---|
| `full` | Standard causal self-attention via Flash Attention (`scaled_dot_product_attention`) |
| `window` | Local context window — each token only attends to the last `w` tokens |
| `dilated` | Strided attention — attends every `d`-th token within a window, capturing long-range structure |
| `global_tokens` | First `k` tokens act as global anchors attending to all positions; remaining heads use a local window |

Example configuration:
```python
head_config = [
    {"type": "full",          "heads": 6,  "param": None},
    {"type": "window",        "heads": 3,  "param": 64},
    {"type": "dilated",       "heads": 2,  "param": (64, 2)},
    {"type": "global_tokens", "heads": 1,  "param": (4, 128)},
]
```

Head configurations can be **switched at runtime** without retraining — existing weights are preserved and transferred automatically.

### GPT-2 Weight Compatibility

The model can load pretrained weights from any GPT-2 variant:

```python
model = GPT.from_pretrained('gpt2')          # 117M
model = GPT.from_pretrained('gpt2-medium')   # 345M
model = GPT.from_pretrained('gpt2-large')    # 774M
model = GPT.from_pretrained('gpt2-xl')       # 1.5B
```

---

## Getting Started

### Requirements
```
torch >= 2.0
transformers
tiktoken
```

Install:
```bash
pip install torch transformers tiktoken
```

### Run
```python
from model import GPT, GPTConfig

# Train from scratch
model = GPT(GPTConfig())

# Or load pretrained GPT-2 weights
model = GPT.from_pretrained('gpt2')
model.eval()
```

---

## Roadmap

This project is actively developed. Planned additions:

- [ ] **Rotary Positional Encoding (RoPE)** — replace learned positional embeddings with RoPE for better length generalization
- [ ] **Additional attention variants** — Sliding Window (Mistral-style), Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- [ ] **Training pipeline** — full training loop with gradient accumulation, cosine LR schedule, and W&B logging
- [ ] **Fine-tuning support** — LoRA / PEFT integration for parameter-efficient fine-tuning
- [ ] **User interface** — interactive UI for text generation and attention visualization
- [ ] **Attention visualization** — heatmaps to inspect which heads specialize in which attention pattern
- [ ] **Benchmarking** — perplexity and downstream task evaluation across head configurations

---

## Project Status

🟡 **Work in progress** — core architecture is complete and functional. Roadmap features are actively being implemented.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al.
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al. (window & dilated attention)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
