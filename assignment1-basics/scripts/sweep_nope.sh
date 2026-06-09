#!/bin/bash
# RoPE vs NoPE ablation
# 题目: Problem (no_pos_emb)
# 时间预算 1 H100 hr

COMMON_ARGS="
    --train_data data/TinyStoriesV2-GPT4-train.npy
    --val_data   data/TinyStoriesV2-GPT4-valid.npy
    --data_dtype uint16
    --vocab_size 10000 --context_length 256
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344
    --batch_size 64 --total_steps 2000
    --warmup_steps 200 --cosine_steps 2000
    --weight_decay 0.05 --grad_clip 1.0
    --eval_interval 500 --eval_batches 20
    --log_interval 25
    --lr_max 1e-2 --lr_min 1e-3
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-nope
    --device cuda:0
"

# 1. RoPE baseline (theta=10000)
echo "=========================================="
echo "=== RoPE baseline (lr=1e-2) ==="
echo "=========================================="
python -m cs336_basics.train $COMMON_ARGS \
    --rope_theta 10000.0 \
    --ckpt_dir runs/rope_lr1e-2 \
    --wandb_run_name rope_lr1e-2 \
    || echo "  [warn] rope failed"

# 2. NoPE (theta=0 → 不建 RoPE)
echo ""
echo "=========================================="
echo "=== NoPE (lr=1e-2) ==="
echo "=========================================="
python -m cs336_basics.train $COMMON_ARGS \
    --rope_theta 0 \
    --ckpt_dir runs/nope_lr1e-2 \
    --wandb_run_name nope_lr1e-2 \
    || echo "  [warn] nope failed"

echo "=== nope ablation done ==="
