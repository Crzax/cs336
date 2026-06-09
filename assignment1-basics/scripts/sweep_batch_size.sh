#!/bin/bash
# Batch size 扫描实验
# 题目: 7.2.X Problem (batch_size_experiment)
#
# 设计决策：
# - 7 个 batch size 跨越 3 个数量级（含 64, 128）
# - 全部跑 3000 步，看趋势够用
# - lr 用 sqrt-scaling: lr = 1e-2 * sqrt(B/64)
# - cosine_steps == total_steps，标准 polishing
# - warmup_steps 按 batch 大小相应调整（小 batch 噪声大、需要更长 warmup）
# - || 让 OOM 不中断后续 run

COMMON_ARGS="
    --train_data data/TinyStoriesV2-GPT4-train.npy
    --val_data   data/TinyStoriesV2-GPT4-valid.npy
    --data_dtype uint16
    --vocab_size 10000 --context_length 256
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344
    --total_steps 3000 --cosine_steps 3000
    --weight_decay 0.05 --grad_clip 1.0
    --eval_interval 300 --eval_batches 20
    --log_interval 25
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-batch
    --device cuda:0
"

# 用 Python 算 sqrt scaling 的 lr（避免 bash 浮点数计算）
declare -A LR_MAP=(
    [1]=1.25e-3
    [8]=3.5e-3
    [32]=7.1e-3
    [64]=1.0e-2
    [128]=1.4e-2
    [512]=2.8e-2
    [2048]=5.7e-2
)

for B in 1 8 32 64 128 512 2048; do
    LR=${LR_MAP[$B]}
    LRMIN=$(python -c "print($LR * 0.1)")

    # 小 batch 噪声大，warmup 多一点
    if [ "$B" -le 8 ]; then
        WARMUP=300
    elif [ "$B" -le 64 ]; then
        WARMUP=200
    else
        WARMUP=100
    fi

    NAME="bs${B}_lr${LR}"
    echo ""
    echo "=========================================="
    echo "=== Running B=$B  lr=$LR  warmup=$WARMUP ==="
    echo "=========================================="
    python -m cs336_basics.train $COMMON_ARGS \
        --batch_size $B \
        --lr_max $LR --lr_min $LRMIN \
        --warmup_steps $WARMUP \
        --ckpt_dir runs/$NAME \
        --wandb_run_name $NAME \
        || echo "  [warn] B=$B (lr=$LR) failed (likely OOM or divergence), continuing..."
done

echo "=== batch sweep done ==="
