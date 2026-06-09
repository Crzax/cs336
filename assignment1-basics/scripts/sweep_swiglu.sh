#!/bin/bash
# SwiGLU vs SiLU FFN ablation
# 题目: Problem (swiglu_ablation)
# 参数量大致相等：
#   SwiGLU: 3 * 512 * 1344 = 2,064,384 (per layer)
#   SiLU:   2 * 512 * 2048 = 2,097,152 (per layer)，差距 1.6%
#
# 时间预算 1 H100 hr，2 个 run 各 ~10 min

COMMON_ARGS="
    --train_data data/TinyStoriesV2-GPT4-train.npy
    --val_data   data/TinyStoriesV2-GPT4-valid.npy
    --data_dtype uint16
    --vocab_size 10000 --context_length 256
    --d_model 512 --num_layers 4 --num_heads 16
    --batch_size 64 --total_steps 2000
    --warmup_steps 200 --cosine_steps 2000
    --weight_decay 0.05 --grad_clip 1.0
    --eval_interval 500 --eval_batches 20
    --log_interval 25
    --lr_max 1e-2 --lr_min 1e-3
    --rope_theta 10000.0
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-swiglu
    --device cuda:0
"

# 1. SwiGLU baseline (3 矩阵, d_ff=1344)
echo "=========================================="
echo "=== SwiGLU baseline (d_ff=1344) ==="
echo "=========================================="
python -m cs336_basics.train $COMMON_ARGS \
    --ffn_type swiglu --d_ff 1344 \
    --ckpt_dir runs/swiglu_baseline \
    --wandb_run_name swiglu_baseline \
    || echo "  [warn] swiglu failed"

# 2. SiLU FFN (2 矩阵, d_ff=2048 = 4*d_model)
echo ""
echo "=========================================="
echo "=== SiLU FFN (d_ff=2048) ==="
echo "=========================================="
python -m cs336_basics.train $COMMON_ARGS \
    --ffn_type silu --d_ff 2048 \
    --ckpt_dir runs/silu_baseline \
    --wandb_run_name silu_baseline \
    || echo "  [warn] silu failed"

echo "=== swiglu ablation done ==="
