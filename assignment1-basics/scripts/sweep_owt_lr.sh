#!/bin/bash
# OWT lr sweep —— Phase 1
# 目标: 找到 OWT 上最优 lr (TS 上是 1e-2，OWT 因 vocab 翻倍可能要降)
# 每 run 1500 步约 8 min, 4 个 run 共 ~30 min

cd /mnt/cephfs/user_crzaxchen/336/assignment1-basics

COMMON_ARGS="
    --train_data data/owt_train.npy
    --val_data   data/owt_valid.npy
    --data_dtype uint16
    --vocab_size 32000 --context_length 256
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344
    --rope_theta 10000.0
    --batch_size 64 --total_steps 1500
    --warmup_steps 200 --cosine_steps 1500
    --weight_decay 0.05 --grad_clip 1.0
    --eval_interval 500 --eval_batches 20
    --log_interval 25
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-owt-lr
    --device cuda:0
"

for LR in 7.5e-3 1.5e-2; do
    LRMIN=$(python -c "print($LR * 0.1)")
    NAME="owt_sweep_lr${LR}"
    echo ""
    echo "=========================================="
    echo "=== OWT lr sweep: lr=$LR ==="
    echo "=========================================="
    python -m cs336_basics.train $COMMON_ARGS \
        --lr_max $LR --lr_min $LRMIN \
        --ckpt_dir runs/$NAME \
        --wandb_run_name $NAME \
        || echo "  [warn] lr=$LR failed"
done

echo "=== owt lr sweep done ==="
