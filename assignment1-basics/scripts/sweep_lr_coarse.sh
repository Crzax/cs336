#!/bin/bash
# 在项目根目录执行：
#   cd /home/crzaxchen/336/assignment1-basics
#   ./cs336_basics/runs/sweep_lr_coarse.sh


COMMON_ARGS="
    --train_data data/TinyStoriesV2-GPT4-train.npy
    --val_data   data/TinyStoriesV2-GPT4-valid.npy
    --data_dtype uint16
    --vocab_size 10000 --context_length 256
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344
    --batch_size 64 --total_steps 2000
    --warmup_steps 200 --cosine_steps 2000
    --grad_clip 1.0 --weight_decay 0.1
    --eval_interval 500 --eval_batches 20
    --log_interval 25
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1-lr-coarse
    --device cuda:0
"

for LR in 1.5e-2; do
    NAME="coarse_lr${LR}"
    echo "=== Running $NAME ==="
    python -m cs336_basics.train $COMMON_ARGS \
        --lr_max $LR --lr_min $(python -c "print($LR * 0.1)") \
        --ckpt_dir runs/$NAME \
        --wandb_run_name $NAME \
        || echo "  [warn] $NAME exited non-zero (likely diverged), continuing..."
done

echo "=== sweep done ==="
