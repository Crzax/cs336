#!/bin/bash
# OWT 模型推理
# 用法: bash scripts/infer_owt.sh "Your prompt here" [LR_TAG]

PROMPT=${1:-"The economic theory of marginal utility states that"}
LR_TAG=${2:-1.5e-2}
CKPT=runs/owt_lr${LR_TAG}/final.pt

cd /mnt/cephfs/user_crzaxchen/336/assignment1-basics

python -m cs336_basics.infer \
    --ckpt   "$CKPT" \
    --vocab  vocab_owt.json \
    --merges merges_owt.txt \
    --prompt "$PROMPT" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.9 \
    --vocab_size 32000 --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --rope_theta 10000.0 \
    --eot_token "<|endoftext|>" \
    --device cuda:0 \
    --seed 42
