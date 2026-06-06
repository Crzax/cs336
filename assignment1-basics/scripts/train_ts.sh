# v1: 327M tokens, val_loss 1.464 (just miss target 1.45)
# python -m cs336_basics.train \
#     --train_data data/TinyStoriesV2-GPT4-train.npy \
#     --val_data   data/TinyStoriesV2-GPT4-valid.npy \
#     --data_dtype uint16 \
#     --vocab_size 10000 --context_length 256 \
#     --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
#     --batch_size 64 --total_steps 20000 \
#     --lr_max 1e-2 --lr_min 1e-3 \
#     --warmup_steps 500 --cosine_steps 20000 \
#     --weight_decay 0.1 --grad_clip 1.0 \
#     --eval_interval 1000 --eval_batches 50 \
#     --log_interval 100 \
#     --ckpt_dir runs/exp_final_lr1e-2 \
#     --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1 \
#     --wandb_run_name exp_final_lr1e-2 \
#     --device cuda:0

# v2 (FAILED): cosine_steps > total_steps 反而退步到 1.573
# 教训：末段低 lr 是"polishing"，不能跳过。cosine_steps 应该 == total_steps
# python -m cs336_basics.train \
#     --train_data data/TinyStoriesV2-GPT4-train.npy \
#     --val_data   data/TinyStoriesV2-GPT4-valid.npy \
#     --data_dtype uint16 \
#     --vocab_size 10000 --context_length 256 \
#     --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
#     --batch_size 64 --total_steps 20000 \
#     --lr_max 1e-2 --lr_min 1e-3 \
#     --warmup_steps 500 --cosine_steps 28000 \
#     --weight_decay 0.1 --grad_clip 1.0 \
#     --eval_interval 1000 --eval_batches 50 \
#     --log_interval 100 \
#     --ckpt_dir runs/exp_final_v2 \
#     --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1 \
#     --wandb_run_name exp_final_v2_cosine28k \
#     --device cuda:0

# v3: lr=1.5e-2 + wd=0.05 + 25k steps → val_loss 1.40 ✓ (passed target 1.45)
# python -m cs336_basics.train \
#     --train_data data/TinyStoriesV2-GPT4-train.npy \
#     --val_data   data/TinyStoriesV2-GPT4-valid.npy \
#     --data_dtype uint16 \
#     --vocab_size 10000 --context_length 256 \
#     --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
#     --batch_size 64 --total_steps 25000 \
#     --lr_max 1.5e-2 --lr_min 1.5e-3 \
#     --warmup_steps 500 --cosine_steps 25000 \
#     --weight_decay 0.05 --grad_clip 1.0 \
#     --eval_interval 1000 --eval_batches 50 \
#     --log_interval 100 \
#     --ckpt_dir runs/exp_final_v3 \
#     --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1 \
#     --wandb_run_name exp_final_v3_lr1.5e-2_wd0.05_25k \
#     --device cuda:0

# 观察：v1 (lr=1e-2) 在同步数下 val_loss 比 v3 (lr=1.5e-2) 更低
# v3 只靠多 5k 步反超。所以最优组合是 v1 的 lr + v3 的其他改动。

# v4: 路径 A 修正 —— 保留 v1 的 lr，套用 v3 的 wd + 长训练
python -m cs336_basics.train \
    --train_data data/TinyStoriesV2-GPT4-train.npy \
    --val_data   data/TinyStoriesV2-GPT4-valid.npy \
    --data_dtype uint16 \
    --vocab_size 10000 --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --batch_size 64 --total_steps 25000 \
    --lr_max 1e-2 --lr_min 1e-3 \
    --warmup_steps 500 --cosine_steps 25000 \
    --weight_decay 0.05 --grad_clip 1.0 \
    --eval_interval 1000 --eval_batches 50 \
    --log_interval 100 \
    --ckpt_dir runs/exp_final_v4 \
    --wandb --wandb_entity sglang-vllm-pku --wandb_project cs336-a1 \
    --wandb_run_name exp_final_v4_lr1e-2_wd0.05_25k \
    --device cuda:0

