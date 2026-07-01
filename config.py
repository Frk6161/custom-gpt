# from dataclasses import dataclass

# @dataclass
# class GPTConfig:
#     block_size: int = 1024
#     vocab_size: int = 50257
#     n_layer: int = 12
#     n_head: int = 12
#     n_embd: int = 768
#     head_config: list = None

#     def __post_init__(self):
#         if self.head_config is None:
#             self.head_config = [{"type": "full", "heads": self.n_head, "param": None}]

# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json
import os

#  Modell‑Konfiguration (ersetzt GPTConfig) 
@dataclass
class ModelConfig:
    # Basisparameter
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Attention‑Konfiguration (gemischte Heads)
    head_config: List[Dict[str, Any]] = field(default_factory=list)

    # Optionale Features
    rope: bool = False    # RoPE statt absoluter Position
    swiglu: bool = True   
    qk_norm: bool = True
    kv_cache: bool = False  # KV‑Cache für schnelle Generierung
    sliding_window_size: Optional[int] = None  # für Sliding Window Attention

    def __post_init__(self):
        if not self.head_config:
            self.head_config = [{"type": "full", "heads": self.n_head, "param": None}]

#  Training‑Konfiguration 
@dataclass
class TrainingConfig:
    total_batch_size: int = 524288   # 2^19
    micro_batch_size: int = 4        # B
    sequence_length: int = 1024      # T
    grad_accum_steps: int = 128      # wird automatisch berechnet
    max_steps: int = 50
    warmup_steps: int = 10
    max_lr: float = 6e-4
    min_lr_factor: float = 0.1       # min_lr = max_lr * min_lr_factor
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "float16"           # "float32", "float16", "bfloat16"
    compile_model: bool = True

    log_every: int = 10
    val_every: int = 25
    val_samples: int = 5
    log_gpu_mem: bool = True

    resume_checkpoint: str = None


    def __post_init__(self):
        # Berechne gradient accumulation steps, falls nicht direkt gegeben
        if self.grad_accum_steps == 128:  
            total = self.total_batch_size
            micro = self.micro_batch_size * self.sequence_length
            self.grad_accum_steps = total // micro
            assert total % micro == 0, "total_batch_size muss durch micro_batch_size * seq_len teilbar sein"

#  Optimizer‑Konfiguration 
@dataclass
class OptimizerConfig:
    optimizer_type: str = "AdamW"    # "AdamW", "SGD", ...
    learning_rate: float = 6e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool = True               # fused AdamW verwenden, falls verfügbar

#  RoPE‑Konfiguration (vorbereitet) 
@dataclass
class RoPEConfig:
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None   # z.B. {"type": "linear", "factor": 2.0}

#  KV‑Cache‑Konfiguration 
@dataclass
class KVCacheConfig:
    enabled: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048

#  Sliding Window Attention Config 
@dataclass
class SlidingWindowConfig:
    enabled: bool = False
    window_size: int = 4096

#  Zentrale Konfiguration (hält alle Teil‑Configs) 
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    rope: Optional[RoPEConfig] = None
    kv_cache: Optional[KVCacheConfig] = None
    sliding_window: Optional[SlidingWindowConfig] = None

    @classmethod
    def from_yaml(cls, base_path: str):
        """
        Lädt alle YAML‑Dateien aus dem Ordner `base_path` (z.B. "configs/").
        Erwartet Dateinamen: model_config.yaml, training_config.yaml, optimizer_config.yaml,
        rope_config.yaml, kv_cache_config.yaml, sliding_window_config.yaml.
        Nicht vorhandene optionale Dateien werden ignoriert.
        """
        def load_yaml(filename):
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            return None

        # Modell laden (Pflicht)
        model_dict = load_yaml("model_config.yaml")
        if model_dict is None:
            raise FileNotFoundError(f"model_config.yaml nicht in {base_path}")
        model_cfg = ModelConfig(**model_dict)

        # Training laden (Pflicht)
        train_dict = load_yaml("training_config.yaml")
        if train_dict is None:
            raise FileNotFoundError(f"training_config.yaml nicht in {base_path}")
        train_cfg = TrainingConfig(**train_dict)

        # Optimizer laden (Pflicht)
        opt_dict = load_yaml("optimizer_config.yaml")
        if opt_dict is None:
            raise FileNotFoundError(f"optimizer_config.yaml nicht in {base_path}")
        opt_cfg = OptimizerConfig(**opt_dict)

        # Optionale Konfigurationen
        rope_dict = load_yaml("rope_config.yaml")
        rope_cfg = RoPEConfig(**rope_dict) if rope_dict else None

        kv_dict = load_yaml("kv_cache_config.yaml")
        kv_cfg = KVCacheConfig(**kv_dict) if kv_dict else None

        sw_dict = load_yaml("sliding_window_config.yaml")
        sw_cfg = SlidingWindowConfig(**sw_dict) if sw_dict else None

        return cls(
            model=model_cfg,
            training=train_cfg,
            optimizer=opt_cfg,
            rope=rope_cfg,
            kv_cache=kv_cfg,
            sliding_window=sw_cfg
        )

    @classmethod
    def from_json(cls, base_path: str):
        """Analog zu from_yaml, aber für JSON‑Dateien."""
        def load_json(filename):
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None

        model_dict = load_json("model_config.json")
        if model_dict is None:
            raise FileNotFoundError(f"model_config.json nicht in {base_path}")
        model_cfg = ModelConfig(**model_dict)

        train_dict = load_json("training_config.json")
        if train_dict is None:
            raise FileNotFoundError(f"training_config.json nicht in {base_path}")
        train_cfg = TrainingConfig(**train_dict)

        opt_dict = load_json("optimizer_config.json")
        if opt_dict is None:
            raise FileNotFoundError(f"optimizer_config.json nicht in {base_path}")
        opt_cfg = OptimizerConfig(**opt_dict)

        rope_dict = load_json("rope_config.json")
        rope_cfg = RoPEConfig(**rope_dict) if rope_dict else None
        kv_dict = load_json("kv_cache_config.json")
        kv_cfg = KVCacheConfig(**kv_dict) if kv_dict else None
        sw_dict = load_json("sliding_window_config.json")
        sw_cfg = SlidingWindowConfig(**sw_dict) if sw_dict else None

        return cls(
            model=model_cfg,
            training=train_cfg,
            optimizer=opt_cfg,
            rope=rope_cfg,
            kv_cache=kv_cfg,
            sliding_window=sw_cfg
        )