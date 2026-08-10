from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    train_bin: Path
    valid_bin: Path
    model_output: Path


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    d_model: int = 1024
    d_ff: int = 3072
    num_layers: int = 24
    num_heads: int = 16
    rope_theta: float = 10000.0


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"
    # 原始配置为 train_batch_size=128 / gradient_accumulation_steps=1，
    # 是为 8xB200(180GB) 设计的。本机是 8xH20(95GB)，显存约为一半，
    # batch=128 时 logits 张量 [128, 512, 50257] 在 cross_entropy 里转fp32
    # 需要 ~13GB，加上各层激活会OOM。
    # 这里改为 batch=32 + grad_accum=4，有效 batch 仍为 32*4=128，
    # 每步梯度在数学上与原配置等价，只是拆成4 次前向/反向以降低峰值显存。
    train_batch_size: int = 32
    eval_batch_size: int = 32
    train_steps: int = 16_384
    gradient_accumulation_steps: int = 4
    compile: bool = True
    eval_iterations: int = 1_000
    eval_interval: int = 2_000
    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 6e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 20
    save_checkpoints: bool = False


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
