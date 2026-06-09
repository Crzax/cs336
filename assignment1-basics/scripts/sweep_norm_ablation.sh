#!/bin/bash
# RMSNorm ablation 实验
# 题目: Problem (layer_norm_ablation)
#
# 实验设计:
# 1. 第一阶段 - 用之前的最优 lr (1e-2) 训不带 RMSNorm 的模型 → 预期发散
# 2. 第二阶段 - 一组逐渐降低的 lr，找能稳定的最大值
# 3. 时间预算: 1 H100 hr，所以每个 run 只跑 2000 步
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
    --no_rmsnorm
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-norm-ablation
    --device cuda:0
"

# Stage 1 + 2 一起扫
# lr=1e-2 是 v4 的最优 lr，预期不带 RMSNorm 时发散
# 然后逐次降低 5 倍找稳定上界
for LR in 1e-2 3e-3 1e-3 3e-4 1e-4; do
    LRMIN=$(python -c "print($LR * 0.1)")
    NAME="nornmsnorm_lr${LR}"
    echo ""
    echo "=========================================="
    echo "=== Running no_rmsnorm  lr=$LR ==="
    echo "=========================================="
    python -m cs336_basics.train $COMMON_ARGS \
        --lr_max $LR --lr_min $LRMIN \
        --ckpt_dir runs/$NAME \
        --wandb_run_name $NAME \
        || echo "  [warn] lr=$LR failed (likely diverged), continuing..."
done

# 最后再跑一个**带** RMSNorm 的 lr=1e-2 作为对照
echo ""
echo "=========================================="
echo "=== Running WITH RMSNorm baseline lr=1e-2 ==="
echo "=========================================="
python -m cs336_basics.train \
    --train_data data/TinyStoriesV2-GPT4-train.npy \
    --val_data   data/TinyStoriesV2-GPT4-valid.npy --data_dtype uint16 \
    --vocab_size 10000 --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --batch_size 64 --total_steps 2000 \
    --warmup_steps 200 --cosine_steps 2000 \
    --weight_decay 0.05 --grad_clip 1.0 \
    --eval_interval 500 --eval_batches 20 --log_interval 25 \
    --lr_max 1e-2 --lr_min 1e-3 \
    --ckpt_dir runs/withrmsnorm_lr1e-2 \
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-norm-ablation \
    --wandb_run_name withrmsnorm_lr1e-2 \
    --device cuda:0

echo "=== norm ablation done ==="
