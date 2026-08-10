"""在 C4 100 domains 验证集上评估已训练模型的 loss。

背景：scripts/train.py 用 logger.info 输出 "Estimated validation loss"，但训练脚本
没有调用 logging.basicConfig()，Python logging 默认级别是 WARNING，导致 INFO 日志
被丢弃、验证 loss 没有留下记录。本脚本直接加载保存好的 model.pt 重新评估。

评估方式与 train.py 的 estimate_dev_loss 保持一致：
  - 用 cs336_basics.data.get_batch 随机采样 batch（与训练同一套取数逻辑）
  - 跑 eval_iters 次，对 cross_entropy 取平均
  - bfloat16 autocast，与训练时的 dtype 一致

用法:
  uv run python scripts/eval_model.py \
      --model-path data/output/your_data \
      --valid-bin data/tokenized_paloma_c4_100_domains_validation.bin

可选：同时在训练集上评估，用于观察过拟合程度（train loss 远低于 valid loss 即过拟合）:
  uv run python scripts/eval_model.py \
      --model-path data/output/your_data \
      --valid-bin data/tokenized_paloma_c4_100_domains_validation.bin \
      --train-bin data/tokenized/data.bin
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM


@torch.no_grad()
def estimate_loss(
    model: BasicsTransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    eval_iters: int,
    device: str,
    context_length: int,
    dtype: torch.dtype,
    desc: str,
) -> float:
    """在给定数据集上估计平均 cross-entropy loss。

    与 train.py 的 estimate_dev_loss 逻辑一致：随机采样 eval_iters 个 batch 取平均。
    """
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    for k in tqdm(range(eval_iters), desc=desc):
        batch_x, batch_y = get_batch(
            dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        with amp_ctx:
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[k] = loss.item()

    return losses.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="模型目录（含 model.pt 和 model_config.json）")
    parser.add_argument("--valid-bin", required=True, help="C4 100 domains 验证集 .bin")
    parser.add_argument("--train-bin", default=None, help="可选：训练集 .bin，用于对比观察过拟合")
    parser.add_argument("--batch-size", type=int, default=32, help="评估 batch size")
    parser.add_argument("--eval-iters", type=int, default=1000, help="评估迭代次数（与训练配置一致）")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="固定随机采样，保证可复现")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    print(f"加载模型: {args.model_path}")
    model = BasicsTransformerLM.from_pretrained(args.model_path)
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,} ({n_params / 1e6:.1f}M)")

    dtype = torch.bfloat16

    # 验证集评估
    valid_data = np.memmap(args.valid_bin, dtype=np.uint16, mode="r")
    print(f"\n验证集: {len(valid_data):,} tokens")
    valid_loss = estimate_loss(
        model=model,
        dataset=valid_data,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        device=args.device,
        context_length=args.context_length,
        dtype=dtype,
        desc="Eval (C4 valid)",
    )

    print("\n" + "=" * 60)
    print(f"C4 100 domains validation loss : {valid_loss:.4f}")
    print(f"C4 100 domains perplexity      : {math.exp(valid_loss):.2f}")
    print("=" * 60)

    # 可选：训练集评估，用于判断过拟合
    if args.train_bin:
        train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
        print(f"\n训练集: {len(train_data):,} tokens")
        train_loss = estimate_loss(
            model=model,
            dataset=train_data,
            batch_size=args.batch_size,
            eval_iters=args.eval_iters,
            device=args.device,
            context_length=args.context_length,
            dtype=dtype,
            desc="Eval (train)",
        )
        print("\n" + "=" * 60)
        print(f"train loss                     : {train_loss:.4f}")
        print(f"train perplexity               : {math.exp(train_loss):.2f}")
        print(f"valid - train (过拟合 gap): {valid_loss - train_loss:.4f}")
        print("=" * 60)
        if valid_loss - train_loss > 1.0:
            print("\n[警告] 验证 loss 明显高于训练 loss，模型过拟合训练数据。")
            print("       通常意味着训练数据量不足（epoch 数过多），建议扩充数据。")


if __name__ == "__main__":
    main()
