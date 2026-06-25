# Memory Profiling

硬件: 单卡 H20 (95 GiB)。模型: xl (d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
vocab=10000。warmup=3, 记录 steps=3。

profiling 入口: `benchmarking_script.py --memory_profile`，复用已有的模型尺寸参数、
`--mixed_precision`、`--run_type`。forward-only 时把前向包在 `torch.no_grad()` 里得到
真正的推理画像（不保留激活）。快照用 https://pytorch.org/memory_viz 的
"Active Memory Timeline" 查看。

## (a) Timelines（fp32, batch=4）

**Forward (inference, ctx=128 / 2048)** — 见 `mem/forward_timeline.png`
总占用基本压在权重底座（≈13 GiB）附近，没有持续爬升的大山；顶部是一排细密波纹，
每个波峰≈一层 Transformer 的瞬态激活（Q/K/V、注意力分数、MLP 中间值），算完即释放。

**Full training step (ctx=128)** — 见 `mem/full_timeline.png`
每个 step 呈"爬坡 → 下坡 → 末尾小台阶"的三段结构，峰值 ≈56 GiB。

**能否分辨阶段：能。** forward 阶段激活单调累积（要留给反向）爬到全局峰值；backward
阶段激活被逐层消费、曲线下降；末尾抬升的小台阶是 optimizer（AdamW 首次分配 m/v 动量）。
最底部一直不释放的 ≈13–14 GiB 底座 = 权重 + 梯度 + 优化器状态。forward-only 没有这座
大山，峰值远低于 full step。

## (b) Peak memory (GiB, fp32)

| context length | forward (inference) | full training step |
| -------------- | ------------------- | ------------------ |
| 128            | 12.81               | 56.20              |
| 2048           | 21.20     | OOM (>95)          |

> ctx=2048 full、fp32 即使 `batch=1` 也 OOM。瓶颈在朴素 attention 物化的
> `(B, H, L, L)` 张量：B=1, H=32, L=2048, fp32 下 **一份 = 32·2048²·4B = 512 MiB/层**，
> scores + softmax weights 两份就是 **≈1 GiB/层**。反向自最后一层往前算，这些张量需保留
> 到对应层的反向完成，峰值期 32 层共约 **30+ GiB**；再加常驻的权重/梯度/AdamW m,v
> （3.4B × 4B × 4 ≈ 51 GiB）和其他激活，总账冲过 95 GiB。这正是后续 FlashAttention /
> 激活重计算要解决的问题（消掉 `L²` 项）。

## (c) Mixed precision

| run               | ctx  | fp32         | bf16 (autocast) | Δ           |
| ----------------- | ---- | ------------ | --------------- | ----------- |
| forward           | 128  | 12.81        | 19.04           | **+6.23**   |
| forward           | 2048 | 21.20       | 25.25           | **+4.05**          |
| full              | 128  | 56.20        | 60.48           | **+4.28**   |
| full              | 2048 | OOM (>95)    | **93.19**       | **bf16 才能跑** |

**结论（2-3 句）：** 混合精度对峰值显存的影响**有限且依赖上下文长度**。在 ctx=128，
autocast 仍保留 FP32 主权重并额外缓存一份 BF16 权重副本（≈6.8 GiB），而激活本身很小，
**净效果反而是变大**；在 ctx=2048，朴素 attention 的 `L×L` 激活成为显存主体，BF16 把它
减半的收益超过权重副本开销，bf16 能跑完而 fp32 同配置 OOM。整体上节省幅度不大——
因为参数、梯度和 AdamW m/v 在 autocast 下仍是 FP32，这才是 full step 的显存大头。

### 为什么会出现"开混合精度反而更大"的反常现象

关键在于 **`torch.autocast` 并不是把模型转成 BF16**，它保留 FP32 主权重，只在算子内部
把权重和激活**临时 cast 成 BF16** 去做 matmul，并为避免重复 cast 而**缓存一份 BF16 权重
副本**。所以 autocast 的显存账本是：

```
权重账: FP32 master (4B/param)  +  BF16 cached copy (2B/param)  =  6B/param
        ≈ 12.8 GiB + 6.8 GiB ≈ 19.6 GiB
激活账: 中间张量从 FP32 减半到 BF16  ≈ 省一半
优化器: AdamW m, v 仍是 FP32，不变
梯度: 主梯度仍是 FP32，不变
```

于是有两股相反的力：

- **多出 ≈+6.8 GiB**：BF16 权重副本（固定开销）。
- **激活减半**：随 ctx 长度增长，收益越大。

哪个占上风看你的 workload：

| | ctx=128 | ctx=2048 |
| --- | --- | --- |
| 激活规模 | 很小 | 主体（`L×L` 是 256×增长） |
| BF16 副本开销 | 仍然 +6.8 GiB | 仍然 +6.8 GiB |
| 净效果 | **多出的副本 > 省的激活 → 变大** | **省的激活 > 多出的副本 → 变小** |

这就是为什么 ctx=128 上 bf16 forward 比 fp32 大了 6.2 GiB（几乎就是那份 BF16 副本），
而在 ctx=2048 full 上 bf16 反而是唯一能跑完的——它把每层 `L×L` 注意力激活从 fp32 减到
bf16（约 1 → 0.5 GiB/层 × 32 层 ≈ 省 16 GiB），刚好让峰值从超过 95 GiB 落到 93.19 GiB。

另一个佐证：fp32 OOM 时报错栈停在 `softmax → torch.exp(rescaled_input)`，这一步需要
为 `L×L` 临时再申请一块 512 MiB。两边的常驻账本其实非常接近（93 GiB vs >95 GiB），
差距就在这种**瞬时尖峰**上——bf16 把尖峰刚好压回了 95 GiB 以内。

## (d) 
Residual stream 张量的形状为 (B, L, d_model)，xl 的 d_model = 2560，fp32 下每元素 4 字节，故大小为 B · L · 2560 · 4 / 1024² MiB：B=1, L=128 时 1.25 MiB；B=1, L=2048 时 20 MiB（batch 是几就乘几）。

## (e)
在 forward-only 快照（xl, ctx=128, batch=4, fp32）降低 Detail 后，可见两类最大 allocation：底部一排 ~100 MiB 的 ghost block，无栈，对应 MLP 各 (d_model=2560, d_ff=10240) 权重矩阵（2560·10240·4B ≈ 100 MiB），它们在 _record_memory_history 启用前由 nn.Linear 构造时分配；顶部一排 ~20 MiB 的瞬态 allocation，栈终止于 mul_Tensor/structured_mul_out，对应 SwiGLU 中 silu(W1·x) * (W3·x) 产生的 (B, L, d_ff)=(4, 128, 10240) 中间激活（4·128·10240·4B = 20 MiB）。

## (f)
第一问：初始51.03GB，forward结束78.02GB，32个block，那就是0.84GB
第二问：最大5个：
op_name        mib    n %
-------------  -----  - —
aten::mul      240.0  6 26.76%
aten::bmm      200.0  4 22.30%
aten::exp      128.0  1 14.27%
aten::div      128.0  1 14.27%
aten::sigmoid  80.0   1 8.92%
总共是896.75MiB

第三问：反向传播的时候梯度，开始是65.33GiB，结束是51.03GiB，净变化是ΔB =(51.03 − 65.33) / 32 ≈ −0.45 GiB/block
梯度G = ΔB + A = −0.45 GiB/block + 0.84 GiB/block = 0.39 GiB/block
预期
attention 4 个投影:  4·d²        = 4·2560²
SwiGLU 3 个 linear:  3·d·d_ff    = 3·2560·10240
一共是0.39GiB符合预期
