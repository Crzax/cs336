# Experiment Log

> 每次跑实验都在最下面追加一段。失败也要记。
> 命名规范：`expNN_short-name`，`runs/expNN_short-name/` 存 ckpt + config.json，wandb run name 同名。

---

## Template（复制下面这段开始记新实验）

### YYYY-MM-DD — expNN_xxx
- **Goal**: 这次想验证什么 / 想看什么
- **Hypothesis**: （可选）你预期的结果
- **Config diff vs baseline**: 只列改了什么（lr=1e-3, batch=64, ...）
- **Command**:
    ```bash
    python -m cs336_basics.train \
        --ckpt_dir runs/expNN_xxx --wandb_run_name expNN_xxx \
        --train_data ... --val_data ... \
        ...
    ```
- **wandb**: <粘贴 run url>
- **Result**:
    - final train_loss / val_loss / ppl
    - wall-clock total time
    - tok/s
- **Observation**: 看到的曲线特征（先快后慢？震荡？平台？发散？）
- **Conclusion / Next**: 由这个结果推出来的下一步

---

## 2026-06-05 — exp00_baseline_smoke
- **Goal**: 跑通整条链路，确认 train + eval + ckpt + wandb 都正常
- **Hypothesis**: 默认配置应能正常下降 loss，无 nan/inf
- **Config diff vs baseline**: 短训练版本（5000 步），用于 smoke test
- **Command**:
    ```bash
    python -m cs336_basics.train \
        --train_data data/TinyStoriesV2-GPT4-train.npy \
        --val_data   data/TinyStoriesV2-GPT4-valid.npy --data_dtype uint16 \
        --vocab_size 10000 --context_length 256 \
        --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size 32 --total_steps 5000 \
        --lr_max 3e-4 --warmup_steps 100 --cosine_steps 5000 \
        --ckpt_dir runs/exp00_baseline --wandb --wandb_run_name exp00_baseline \
        --log_interval 50 --eval_interval 500 --device cuda:0
    ```
- **wandb**: (smoke test，未保留)
- **Result**: pipeline 通畅，loss 从 ~9.2 → ~3.5，无异常
- **Observation**: forward/backward/optimizer/eval/ckpt/wandb 全流程 OK
- **Conclusion / Next**: 进入正式 lr sweep

---

## 2026-06-05 — exp01_coarse_lr_sweep
- **Goal**: 在 4 个数量级范围内 log-spaced 扫 lr，找最优带 + 找发散点
- **Hypothesis**: lr ≈ 1e-3 附近最优；≥1e-2 大概率发散
- **Config diff vs baseline**: 对 lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} 各跑 2000 步短训练
- **Command**: `scripts/sweep_lr_coarse.sh`
    ```bash
    for LR in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
        python -m cs336_basics.train \
            --batch_size 64 --total_steps 2000 \
            --warmup_steps 200 --cosine_steps 2000 \
            --grad_clip 1.0 --weight_decay 0.1 \
            --lr_max $LR --lr_min $(python -c "print($LR * 0.1)") \
            --wandb_project cs336-a1-lr-coarse \
            --wandb_run_name coarse_lr${LR} ...
    done
    ```
- **wandb**: https://wandb.ai/sglang-vllm-pku/cs336-a1-lr-coarse
- **Result** (final train_loss @ step 2000):
    | lr | train_loss | 评价 |
    |---|---|---|
    | 1e-4 | 3.50 | 太慢，欠学 |
    | 3e-4 | 2.70 | 仍偏慢 |
    | 1e-3 | 2.10 | OK |
    | 3e-3 | 1.85 | 好 |
    | 1e-2 | **1.80** | **最佳** |
    | 3e-2 | 1.85 | 略输 1e-2，但仍稳定 |
- **Observation**: 全部 6 个 lr 都没发散——AdamW + grad_clip + cosine 三重保护下，
  常规 transformer 最佳 lr 比预期高很多
- **Conclusion / Next**: hypothesis 错了，1e-3 不是最优。需要继续往上扫找发散点

---

## 2026-06-05 — exp02_extreme_lr_sweep
- **Goal**: 继续往上推 lr，抓 (b) 题需要的 divergent run
- **Hypothesis**: lr ≥ 1e-1 应能看到不稳定 / 发散
- **Config diff vs baseline**: lr ∈ {1e-1, 3e-1}（沿用 coarse sweep 的其他参数）
- **Command**: 同 coarse sweep 脚本，把 LR 列表换成 `1e-1 3e-1`
- **wandb**: 同 cs336-a1-lr-coarse project
- **Result**:
    | lr | train_loss @ 2000 | 现象 |
    |---|---|---|
    | 1e-1 | 2.30 | step 200 (warmup 结束) train_loss spike 到 **7.3**，被 cosine + grad_clip 救回 |
    | 3e-1 | 2.80 | 全程剧烈震荡，多次反弹到 4-6 |
- **Observation**:
    - 1e-1 的 spike 正好出现在 lr 达到峰值的瞬间——典型"edge of stability"
    - 3e-1 在 step 0 起就不稳定，但因 grad_clip 没真正 nan
    - val_loss 都比 lr=1e-2 差 0.7+，工程上不可用
- **Conclusion / Next**: 发散阈值 ≈ 1e-1，最佳 lr ≈ 1e-2，比例 ~1/10。
  锁定 lr=1e-2 进入 full run

---

## 2026-06-05 — exp_final_v1: 第一次 full run
- **Goal**: 用最佳 lr=1e-2 跑满 327M token，目标 val_loss ≤ 1.45
- **Hypothesis**: val_loss 应能落到 1.4 附近
- **Config diff vs baseline**: full run 配置
    - batch_size=64, total_steps=20000 (= 327M tokens)
    - lr_max=1e-2, lr_min=1e-3, warmup=500, cosine_steps=20000
    - weight_decay=0.1, grad_clip=1.0
- **Command**:
    ```bash
    python -m cs336_basics.train \
        --train_data data/TinyStoriesV2-GPT4-train.npy \
        --val_data   data/TinyStoriesV2-GPT4-valid.npy --data_dtype uint16 \
        --vocab_size 10000 --context_length 256 \
        --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size 64 --total_steps 20000 \
        --lr_max 1e-2 --lr_min 1e-3 \
        --warmup_steps 500 --cosine_steps 20000 \
        --weight_decay 0.1 --grad_clip 1.0 \
        --ckpt_dir runs/exp_final_lr1e-2 \
        --wandb_run_name exp_final_lr1e-2 ...
    ```
- **wandb**: https://wandb.ai/sglang-vllm-pku/cs336-a1/runs/hehg96v4
- **Result**:
    - **val_loss 1.464** ⚠️ (target 1.45，差 0.014)
    - train_loss 1.464 → train/val gap ≈ 0（欠拟合，无过拟合）
    - grad_norm 末段 0.59（健康）
    - 44.8 min on H100, 122k tok/s
- **Observation**: 末段 val_loss 缓慢继续下降，疑似收敛但实为 lr 衰减到底
- **Conclusion / Next**: 距目标差 0.014。需要 (1) 拉长训练 或 (2) 调 schedule

---

## 2026-06-06 — exp_final_v2: 拉长 cosine_steps 实验（FAILED ❌）
- **Goal**: 假设 v1 末段平台是 lr 太低导致，拉长 cosine_steps 让末段 lr 高一些
- **Hypothesis**: 末段 lr 从 1e-3 → 2.4e-3，loss 应继续下降
- **Config diff vs v1**: cosine_steps 20000 → **28000**（其他完全相同）
- **Command**:
    ```bash
    --batch_size 64 --total_steps 20000 \
    --lr_max 1e-2 --lr_min 1e-3 \
    --warmup_steps 500 --cosine_steps 28000 \   # 唯一改动
    ...
    ```
- **wandb**: https://wandb.ai/sglang-vllm-pku/cs336-a1/runs/1thm3ixq
- **Result**:
    - **val_loss 1.573** ❌ (比 v1 退步 0.11！)
    - 末端 lr=2.79e-3，grad_norm 0.44（仍健康）
- **Observation**: 全程 val_loss 在 v1 之上，没追平也没反超
- **Conclusion / Next**:
    - **反思**: 末段低 lr 不是"浪费"而是 **polishing**——SGD 噪声小，参数能沉到
      loss landscape 最深处。跳过 polishing 等于在最低点附近震荡
    - **正确做法**: cosine_steps == total_steps，lr_min ≤ lr_max × 0.1
    - **Lesson**: 不要随意改 schedule "形状"，先动 token / lr / wd 这些"维度"

---

## 2026-06-06 — exp_final_v3: 三改一组合改进
- **Goal**: 在 v1 基础上同时调整 lr / wd / steps 三个维度，目标 val_loss ≤ 1.45
- **Hypothesis**: 提 lr + 降 wd + 多 token 应能突破 1.45
- **Config diff vs v1**:
    - lr_max: 1e-2 → **1.5e-2** (lr_min 同比例 → 1.5e-3)
    - weight_decay: 0.10 → **0.05**
    - total_steps: 20000 → **25000** (cosine_steps 同步)
- **Command**:
    ```bash
    --batch_size 64 --total_steps 25000 \
    --lr_max 1.5e-2 --lr_min 1.5e-3 \
    --warmup_steps 500 --cosine_steps 25000 \
    --weight_decay 0.05 --grad_clip 1.0 ...
    ```
- **wandb**: https://wandb.ai/sglang-vllm-pku/cs336-a1/runs/<v3-id>
- **Result**:
    - **val_loss 1.400** ✓ (target 1.45，超额 0.05)
    - 56 min on H100
- **Observation**: v3 全程比 v1 略慢（lr 偏高带来的噪声），但靠多 5k 步反超
- **Conclusion / Next**:
    - 已过线，但同步数对比看 v1 (lr=1e-2) 反而比 v3 (lr=1.5e-2) 略低
    - 怀疑 lr 上调是负贡献，真正起作用的是 wd 和 step 数
    - 下一步：隔离 lr，跑 v4 验证

---

## 2026-06-06 — exp_final_v4: 隔离 ablation (BEST ⭐)
- **Goal**: 验证 v3 改动中 lr=1.5e-2 是否为负贡献
- **Hypothesis**: 真正最优 lr 仍是 1e-2；v3 的提升来自 wd↓ + steps↑
- **Config diff vs v3**:
    - lr_max: 1.5e-2 → **1e-2** (回退)
    - lr_min: 1.5e-3 → **1e-3** (回退)
    - 其他保持 v3 (wd=0.05, steps=25k, cosine=25k, warmup=500)
- **Command**:
    ```bash
    --batch_size 64 --total_steps 25000 \
    --lr_max 1e-2 --lr_min 1e-3 \
    --warmup_steps 500 --cosine_steps 25000 \
    --weight_decay 0.05 --grad_clip 1.0 ...
    ```
- **wandb**: https://wandb.ai/sglang-vllm-pku/cs336-a1/runs/n2ez13ir
- **Result**:
    - **val_loss 1.384** ✓✓ (再比 v3 降 0.016)
    - train_loss 1.325 → train/val gap 0.06（首次出现明显泛化差）
    - 末段 lr=1e-3, grad_norm=0.44
    - 56 min on H100, 122k tok/s
- **Observation**:
    - v4 全程位于 v3 曲线之下 → 验证 lr=1e-2 才是真最优
    - gap=0.06 表明已接近这个超参组合的极限，继续加 token 边际收益急剧下降
- **Conclusion / Next**:
    - **最优配置**: lr=1e-2, wd=0.05, 25k steps, cosine_steps=total_steps
    - **最终 val_loss=1.384，远超目标 1.45**
    - **核心 lesson**: hparam ablation 必须隔离单变量。v3 看似成功的"三改一"
      实际是 1 个负贡献 + 2 个正贡献，没做 v4 会得到错误结论"lr 应该提高"

---

# Cross-cutting Analysis

## (a) LR sweep deliverable

**搜索策略**: 三阶段递进
1. **Coarse log-sweep**（4 个数量级）→ 找最优带
2. **Extreme sweep**（推到崩溃）→ 找发散点 + 给 (b) 题素材
3. **Final ablation**（v1→v2→v3→v4）→ 在最优带做隔离实验

**Learning curves**: 见 wandb project `cs336-a1-lr-coarse` 和 `cs336-a1`

**Final model**: v4 with val_loss = **1.384** ≤ 1.45 ✓

## (b) Edge of Stability

### 观察
- lr ≤ 3e-2: 全程稳定，loss 单调下降
- lr = 1e-2: best, full-run val_loss = 1.384
- lr = 1e-1: warmup 结束瞬间 spike 到 train_loss=7.3（高于初值 9 仅 1.7），
            被 grad_clip + cosine 救回，但 val 仍差 0.7
- lr = 3e-1: 从 step 0 起持续震荡，多次反弹到 4-6

### 推论
- **发散阈值** lr_div ≈ 1e-1
- **最佳 lr** lr_best ≈ 1e-2
- **比例** lr_best / lr_div ≈ **0.1**，与 folk wisdom (1/3 ~ 1/10) 一致

### 工程含义
即便有 grad_clip + cosine schedule + AdamW 三重保护，跨过发散阈值仍会带来：
1. 显著的早期 loss spike（即使最终能恢复）
2. 收敛速度慢得多（同 step 下 val_loss 高 0.5+）
3. 实际"安全"的 lr 上限远低于"绝对发散"的上限——
   不能用"还没 nan"来判断 lr 合适，要看是否平滑下降
