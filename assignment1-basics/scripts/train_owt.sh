#!/bin/bash
# OWT full-run training
# 用法: bash scripts/train_owt.sh [LR]    (默认 1e-2)
#
# 与 TinyStories v4 相同 architecture 和 iterations:
#   - d_model=512, num_layers=4, num_heads=16, d_ff=1344
#   - batch_size=64, total_steps=25000
#   - warmup=500, cosine_steps=25000
# 唯一允许调的: lr (sweep 后填入), 数据集, vocab_size

LR=${1:-1.5e-2}
LRMIN=$(python -c "print($LR * 0.1)")
TAG=$(echo $LR | sed 's/[^a-zA-Z0-9.-]/_/g')

cd /mnt/cephfs/user_crzaxchen/336/assignment1-basics

python -m cs336_basics.train \
    --train_data data/owt_train.npy \
    --val_data   data/owt_valid.npy \
    --data_dtype uint16 \
    --vocab_size 32000 --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --rope_theta 10000.0 \
    --batch_size 64 --total_steps 25000 \
    --lr_max $LR --lr_min $LRMIN \
    --warmup_steps 500 --cosine_steps 25000 \
    --weight_decay 0.05 --grad_clip 1.0 \
    --eval_interval 1000 --eval_batches 50 \
    --log_interval 100 \
    --ckpt_dir runs/owt_lr${TAG} \
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-owt \
    --wandb_run_name owt_lr${TAG} \
    --device cuda:0
