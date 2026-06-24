# mixed_precision_accumulation
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
主要是如果accumulator如果是低精度，这导致了swamping

# benchmarking_mixed_precision
## (a)
the model parameters within the autocast context?
FP32 (autocast 不改变参数本身的存储 dtype)
• the output of the first feed-forward layer (ToyModel.fc1)?
FP16 (Linear/matmul 在 autocast 白名单, 降到 FP16)
• the output of layer norm (ToyModel.ln)?
FP32 (LayerNorm 在 autocast 的 fp32 列表, 为数值稳定保持 FP32)
• the model’s predicted logits?
FP16
• the loss?
FP32
• the model’s gradients?
FP32

## (b)
求均值和方差的累加，方差的平方根和求倒数，归一化除法
换成BF16理论上不需要，因为范围很大，但是精度降低了，尾数只有 7 位，远少于 FP16 的 10 位、FP32 的 23 位）。在一行里累加很多元素求均值/方差时，舍入误差/swamping 仍然存在，甚至比 FP16 更严重。所以实践中（包括 PyTorch autocast 的实现）出于"精度"而非"范围"的考虑，仍然倾向于让 LayerNorm 的 reduction 在 FP32 中累加。

## (c)
硬件: 单卡 H20 (95 GiB), batch=4, context=512, warmup=2, steps=10。
新增 `--mixed_precision` 开关 (BF16 autocast), 关闭时用 `nullcontext` no-op。
计时单位 ms/step (10 步均值), 括号为 BF16 相对 FP32 加速比:

| size   | forward fp32 | forward bf16 | fwd+bwd fp32 | fwd+bwd bf16 | full fp32 | full bf16 |
|--------|-------------:|-------------:|-------------:|-------------:|----------:|----------:|
| small  |        29.21 | 15.36 (1.90×)|        91.19 | 43.00 (2.12×)|     99.00 | 51.64 (1.92×)|
| medium |        93.40 | 38.23 (2.44×)|       275.0  |114.1 (2.41×) |    296.1  |134.3 (2.20×) |
| large  |       200.3  | 75.21 (2.66×)|       588.2  |224.1 (2.62×) |    633.3  |269.2 (2.35×) |
| xl     |       639.3  |174.3 (3.67×) |      1801    |536.1 (3.36×) |   1944    |676.2 (2.87×) |
| 10B    |          OOM |          OOM |          OOM |          OOM |       OOM |       OOM |

(所有运行 std 均在 0.04–1.7 ms, 测量非常稳定; 10B 在单卡 95 GiB 上 fp32/bf16 均 OOM。)

**结论 (2-3 句)**: BF16 混合精度在所有可运行的尺寸上都比 FP32 快, 且加速比随模型增大而上升——
small 约 1.9×, 到 xl 前向已达 3.67×, 因为大模型由矩阵乘主导, BF16 走 Tensor Core 且显存带宽减半, 收益最大;
小模型受 kernel launch / Python 开销稀释, 加速相对小。
同一尺寸内 `forward` 加速 > `full`, 因为 AdamW 优化器步在 FP32 中执行且访存受限, 拉低了整步的平均加速比。