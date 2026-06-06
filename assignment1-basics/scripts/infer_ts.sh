#!/bin/bash
# 用 TinyStories 上训好的 v4 模型生成文本
# 必须从项目根 ..../336/assignment1-basics 执行

CKPT=runs/exp_final_v4/final.pt
PROMPT=${1:-"Once upon a time, there was a little girl named Lily."}

python -m cs336_basics.infer \
    --ckpt   "$CKPT" \
    --vocab  vocab_ts.json \
    --merges merges_ts.txt \
    --prompt "$PROMPT" \
    --max_new_tokens 200 \
    --temperature 0.1 \
    --top_p 0.9 \
    --vocab_size 10000 --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --rope_theta 10000.0 \
    --eot_token "<|endoftext|>" \
    --device cuda:0 \
    --seed 42
