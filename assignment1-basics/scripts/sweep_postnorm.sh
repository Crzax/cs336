#!/bin/bash
# Pre-norm vs Post-norm ablation
# 题目: Problem (pre_norm_ablation)
#
# 时间预算 1 H100 hr：每 run 跑 2000 步约 10 min
# 4 个 run = ~40 min
#
# 实验设计：
# - pre-norm baseline @ lr=1e-2  (复用 v4 最优 lr)
# - post-norm @ lr=1e-2          (预期不稳定/慢/差)
# - post-norm @ lr=3e-3          (降 lr 试稳定)
# - post-norm @ lr=1e-3          (再降一档)
#

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
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-postnorm
    --device cuda:0
"

# 1. Pre-norm baseline
echo "=========================================="
echo "=== pre-norm baseline (lr=1e-2) ==="
echo "=========================================="
python -m cs336_basics.train $COMMON_ARGS \
    --norm_position pre \
    --lr_max 1e-2 --lr_min 1e-3 \
    --ckpt_dir runs/prenorm_lr1e-2 \
    --wandb_run_name prenorm_lr1e-2 \
    || echo "  [warn] pre-norm failed"

# 2-4. Post-norm at decreasing lr
for LR in 1e-2 3e-3 1e-3; do
    LRMIN=$(python -c "print($LR * 0.1)")
    NAME="postnorm_lr${LR}"
    echo ""
    echo "=========================================="
    echo "=== post-norm  lr=$LR ==="
    echo "=========================================="
    python -m cs336_basics.train $COMMON_ARGS \
        --norm_position post \
        --lr_max $LR --lr_min $LRMIN \
        --ckpt_dir runs/$NAME \
        --wandb_run_name $NAME \
        || echo "  [warn] $NAME failed"
done

echo "=== post-norm ablation done ==="
