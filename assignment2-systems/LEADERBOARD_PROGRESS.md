# Leaderboard 优化进度交接（Assignment 2）

## 目标
优化 8B 模型「完整一步训练」(forward + loss + backward + AdamW) 的 wall-clock 时间。
- 约束：不改模型 I/O 行为、BF16 + causal mask、自己实现、从空 cache 起 10 分钟内跑完、打败 10s 基线。
- 官方测试环境：2×B200，batch=2，seq_len=32768，config 见 `cs336_systems/leaderboard.py`。

## 运行环境（重要）
- 跑命令用 `uv run ...`
- H20 是 Hopper：Triton/torch.compile/TMA 都支持，可验证正确性和相对加速；但绝对时间无法与 B200 可比（H20 算力约为 B200 的 ~1/10）。

## 8B config 显存账（记忆用）
- 参数量 ≈ 8.13B（embedding 622M + 独立 lm_head 622M + 34 层 ×202M；**无 tie weight**）。
- 静态（参/梯/优化器 m,v）：当前 AdamW 用 `zeros_like(grad)` → m/v 是 bf16，共 8 字节/参 ≈ 65GB；若 m/v 改 fp32 → 12 字节 ≈ 97GB。
- Logits+CE naive：`[2,32768,151936]` fp32 中间量 ≈ 40GB（#1 的靶子）。
- Attention scores naive：`[2,32,32768,32768]` ≈ 137GB/层（必须用 Flash，非优化而是前提）。
- 逐层激活：不做 checkpointing 约 250~320GB（#6 的靶子）。

---

## TODO 表与状态

| # | 任务 | 优化维度 / 收益来源 | 状态 |
|---|------|------|------|
| 1 | Fused Cross-Entropy + LM Head | **省显存**（compute-bound）：不 materialize `[N,V]` 全量 logits，峰值 `[N,V]→[chunk,V]`（satisfy 40GB→GB 级）；顺带省掉 logits 大张量的 HBM 往返。腾出的显存换更大 batch/seq。 | ✅ **已完成** |
| 2 | Fused AdamW（Triton kernel 逐元素更新） | **提速 + 省 kernel launch**（memory-bound）：6 个 elementwise op 融合成 1 个 kernel，p/m/v/grad 每元素 HBM 只读一次写一次（省 ~5 轮往返），launch 从 ~6 次/张量→1 次/张量。 | ✅ **已完成** |
| 3 | Flash causal early stop（跳过对角线上方全 0 tile） | **提速**（省算力）：causal 下 K tile 全在 Q tile 右上方时结果恒 0，不迭代它。理论省掉约一半 attention FLOPs（下三角只占一半）。 | ✅ **已完成** |
| 4 | Flash L/D tile 分离（非对角块不做索引比较，仅对角块比一次） | **提速**（省无效指令）：非对角块要么全保留要么全跳过，无需逐元素 `q_pos>=k_pos` 比较与 `where`；仅对角块需要 mask。减少 kernel 内分支/比较开销。 | ✅ **已完成** |
| 5 | Triton autotune（tile size / num_warps / num_stages） | **提速**（占用率/流水线调优）：让 Triton 自动搜最优 tile 尺寸、warp 数、pipeline stage，逼近硬件 SM 占用率与访存流水线上限。 | ✅ **已完成** |
| 6 | Activation checkpointing（算力换显存塞下大 seq） | **省显存**（拿算力换）：只存少量 checkpoint，反向时重算中间激活，逐层激活从 250~320GB 大幅下降 → 才塞得下 seq=32768。 | ✅ **已完成** |
| 7 | FSDP 切分静态状态跨 2 卡 + torch.compile + 端到端 full-step 计时 | **省显存 + 提速**（系统级）：参/梯/优化器状态按 2 卡切分（每卡 ~1/2 静态显存）；torch.compile 做算子融合与图优化；集成前 6 项跑端到端计时。 | ✅ **已完成** |

排序逻辑：1、2 独立大头先做；3、4 都改 `opt_kernel.py` 三个 kernel，相邻做；5 等 kernel 就绪统一 autotune；6、7 系统级收尾。

**收益维度速记**：优化只有两个大方向——**省显存**（#1/#6，以及 #7 的切分，目标是「塞得下」）和 **提速**（#2/#3/#4/#5，目标是「跑得快」）。判断该用什么工具，先看负载是 **compute-bound**（算力受限，如 GEMM → 靠 cuBLAS / 少算 FLOPs）还是 **memory-bound**（访存受限，如逐元素 op → 靠 Triton 融合减少 HBM 往返与 launch）。

**工作流**：每项 = 写实现 → 单元测正确性 → 小规模基准看收益。**不要每步都跑 `leaderboard.py`**（满配终极测试，几分钟且极吃显存），留到 #7 集成后再跑，且先用小 rep/warmup。

---

## #1 已完成详情（Fused CE + LM Head）

- 实现文件：`cs336_systems/fused_ce.py`，函数 `fused_linear_cross_entropy(x, weight, targets, chunk_size=4096)`。
- 测试文件：`tests/test_fused_ce.py`。**结果：4 个用例全过（fp32 逐位 + bf16 宽松），显存 naive 3.433GB → fused 2.663GB (1.3x，小规模；真实 V=151936/N=65536 时差距 5~10x)**。

**设计（三句话）**：
1. **分块**：N 个 token 切段，每段只算 `[chunk,V]` 小 logits，算完即弃 → 峰值从 `[N,V]` 降到 `[chunk,V]`。
2. **闭式梯度**：CE 对 logits 梯度 = `softmax - onehot`，不用 autograd。
3. **即时反向**：forward 里顺手算好 `dx=g@W`、`dW+=gᵀ@x` 存入 ctx，backward 只做标量缩放。

**关键点**：GEMM 走 cuBLAS（bf16 tensor core），softmax 升 fp32；`1/N` 提前折进 g；dW fp32 累加最后转 bf16。**#1 没用 Triton**——因为计算 99% 是 GEMM，是 cuBLAS 主场；Triton 价值在融合访存密集碎操作，不是重写 GEMM。

**待集成**：目前模型 `model(labels)` 仍返回全量 logits，真正用上 fused CE 要在 #7 集成阶段改训练步走「隐藏态 → fused loss」，不再 materialize logits。

---

## #2 已完成详情（Fused AdamW）

- 实现文件：`cs336_systems/fused_adamw.py`，类 `FusedAdamW`（接口与参照 `AdamW` 完全一致），核心是 `_adamw_kernel`（Triton `@triton.jit`）。
- 测试文件：`tests/test_fused_adamw.py`。正确性用**多步迭代对齐**参照 AdamW（fp32 逐位 1e-5、bf16 宽松 2e-2、含 weight_decay=0 分支），`main()` 里带耗时基准。

**为什么这里该用 Triton（与 #1 相反）**：AdamW 是纯 **memory-bound**（逐元素、几乎无算术、无 GEMM）。原生 `step()` 对每个参数张量发起 ~6 个独立 elementwise op（`p*=`、`m=…`、`v=…`、`sqrt`、`div`、`p-=`），每个 op = 一次 kernel launch + 一轮 p/m/v/grad 的 HBM 读写。融合成单 kernel 后：每元素只读一次写一次，中间量全在寄存器；kernel launch 从 ~6 次/张量降到 1 次/张量。

**设计（三句话）**：
1. **单 kernel 融合**：m 更新 + v 更新 + 偏置校正 + 解耦权重衰减 + 参数更新，全塞进一个 `_adamw_kernel`，一维网格 `grid=cdiv(numel, BLOCK_SIZE)`，线性偏移访问（要求张量内存连续，非连续时 `.contiguous()` 兜底再写回）。
2. **标量偏置校正在 host 算**：`alpha_t = lr * sqrt(1-beta2^t)/(1-beta1^t)` 是每张量一个标量，在 Python 侧算好当 kernel 参数传入，kernel 里不再 `pow(beta,t)`。
3. **fp32 计算 / 原 dtype 存储**：p/g/m/v 载入后 `.to(tl.float32)` 再算（m/v 底层 bf16，中间用 bf16 会累积误差）；`tl.store` 自动 cast 回指针元素类型，m/v 沿用参照 `zeros_like(grad)` 的 dtype。

**数学与参照严格一致**（`cs336_basics/optimizer.py`）：权重衰减用 `lr`（非 `alpha_t`）、`t` 从 1 起。

**待集成（#7）**：leaderboard 训练步把 `AdamW(model.parameters())` 换成 `FusedAdamW(...)` 即可，接口零改动。

---

## #3 已完成详情（Flash causal early stop）

- 改动文件：`cs336_systems/opt_kernel.py` 三个 kernel（`flash_fwd_kernel` / `flash_bwd_dq_kernel` / `flash_bwd_dkdv_kernel`）。
- 正确性：复用已有 `tests/test_attention.py`（`is_causal=True` 已覆盖前向+反向 Triton），逻辑只删全 0 tile 不改数值，故无需新测；加速看 `benchmark_flash.py`（默认 `is_causal=True`）。

**原理**：causal mask 下 query 只看 key ≤ 自己（注意力矩阵只用下三角）。原实现仍完整遍历所有 K/Q tile，对角线上方那半张全是 `-1e6→exp→0`，纯属白算。early stop = **让被完全 mask 的 tile 根本不进循环**（只砍全 0 tile；对角线上骑跨的那一个 tile 仍走原有逐元素 mask，那是 #4 的靶子）。

**三个 kernel 的切法（关键差异：按谁并行就跳另一维）**：
1. `flash_fwd_kernel` / `flash_bwd_dq_kernel`（按 **query tile** 并行，内层循环 K tile）：本 Q tile 最大 query 下标 `=(qt+1)*Q_TILE-1`，只需算到覆盖它的 K tile → `num_key_tiles = cdiv((qt+1)*Q_TILE, K_TILE)`，循环上界从「全部 K tile」缩到这个值。
2. `flash_bwd_dkdv_kernel`（按 **key tile** 并行，内层循环 Q tile）：反过来，本 K tile 最小 key 下标 `=kt*K_TILE`，`max_q` 小于它的 query tile 全被 mask → **起点** `start_query_tile = (kt*K_TILE)//Q_TILE`，把 Q/dO/L/O 四个 `block_ptr` 的初始 `offset` 直接落到 `start_query_tile*Q_TILE`，循环 `range(start_query_tile, num_query_tiles)`。

Q_TILE==K_TILE 时正好各砍到对角线，平均省约一半 tile。公式是保守的（绝不跳有效元素），Q_TILE≠K_TILE 或尾块也安全。

---

## #4 已完成详情（Flash L/D tile 分离）

- 改动文件：同 `cs336_systems/opt_kernel.py` 三个 kernel。正确性复用 `tests/test_attention.py`（Triton causal 前反向已覆盖），逻辑只把原来"所有 K tile 都跑 mask+where"拆成两段，数值等价。

**原理**：#3 只砍纯 0 tile（对角线上方），剩下的 tile 里其实还分两种：
- **纯下三角非对角块**（本 K tile 最大 key < 本 Q tile 最小 query）：`q>=k` 恒成立，逐元素 mask 100% 是白算的 compare + where。
- **对角块**（Q/K tile 骑跨对角线）：只有这里才真正需要逐元素比较 + `tl.where(mask, Sij, -1e6)`。

分成两段循环后：非对角段完全没有 `arange/>=/where` 指令，编译器生成的就是纯粹的 `dot → exp → dot`，指令更紧、寄存器压力更小；对角段每个 program 只需 1~2 个 tile 走 mask。

**三个 kernel 的切法（依然按谁并行区分对称与否）**：
1. `flash_fwd_kernel` / `flash_bwd_dq_kernel`（按 query tile 并行）：
   - `n_full_tiles = qt*Q_TILE // K_TILE`（下取整）——本 Q tile 最小 query 完全覆盖到的 K tile 数，这些是纯下三角非对角块。
   - `n_total_tiles = cdiv((qt+1)*Q_TILE, K_TILE)`——覆盖到本 Q tile 最大 query 的 K tile 上界，`[n_full, n_total)` 是对角块。
   - 段 1：`for kt in range(n_full_tiles)` 无 mask；段 2：`for kt in range(n_full_tiles, n_total_tiles)` 有 mask。
2. `flash_bwd_dkdv_kernel`（按 key tile 并行，反过来切 Q tile）：
   - `start_query_tile = (kt*K_TILE)//Q_TILE`（#3 保留，跳前面纯上三角块的起点）。
   - `first_full_qt = cdiv((kt+1)*K_TILE, Q_TILE)`——本 K tile 最大 key 严格小于该 Q tile 最小 query 的起点，之后都是纯下三角非对角块。
   - 段 1：`for qt in range(start_query_tile, first_full_qt)` 对角块，有 mask；段 2：`for qt in range(first_full_qt, N_Q_tiles)` 非对角块，无 mask。

**边界验证**（Q_TILE=K_TILE=16 时）：
- fwd/dq `qt=0`：`n_full=0, n_total=1` → 只跑 1 个对角块（正确，第 0 个 Q tile 只跟第 0 个 K tile 交，且骑跨对角线）。
- fwd/dq `qt=1`：`n_full=1, n_total=2` → 段 1 跑 kt=0（下三角），段 2 跑 kt=1（对角）。
- dkdv `kt=0`：`start=0, first_full=1` → 段 1 跑 qt=0（对角），段 2 跑 qt=1..N（下三角）。

Q_TILE≠K_TILE 或尾块也安全：cdiv/floor 除法都是保守的，绝不把对角块划到"无 mask"段。非 causal 时前向/dq 让 `n_full = n_total = cdiv(N_KEYS,K_TILE)`（段 2 空转）；dkdv 让 `first_full_qt = start_query_tile`（段 1 空转），逻辑与原版一致。

---

## #5 已完成详情（Triton autotune）

- 改动文件：`cs336_systems/opt_kernel.py`——文件顶部加 3 组 `triton.Config` 列表；三个 kernel 加 `@triton.autotune(configs=..., key=[...])` 装饰器；host 侧 `FlashAttnFuncTriton` 里 grid 改成 `lambda META: (triton.cdiv(nq, META['Q_TILE_SIZE']), bs)`，删掉显式 `Q_TILE_SIZE/K_TILE_SIZE=...` 传参。
- 正确性复用 `tests/test_attention.py`，逻辑与数值不变。首次调用会自动 benchmark 所有 config、选最快的并缓存。

**为什么需要 autotune**：#3/#4 已经把算法层面能省的都省了，剩下的收益来自**硬件占用率**——同一份 kernel 换不同 (tile, warps, stages) 组合，在不同 (seq_len, d, dtype) 下最优点可能差 2~3 倍。手工挑的 `_pick_tile` 只是 shared memory 兜底，没考虑寄存器压力、warp 并发、async pipeline 深度。autotune 让编译器/运行时替我们做这个 sweep。

**设计要点（五句话）**：
1. **key 选 `[N_QUERIES, N_KEYS, D, is_causal]`**——决定最优 config 的所有维度。dtype 不用放（Triton 内部对 bf16/fp32 各自 specialize）；`stride_*` 更不用（不影响性能选择）。
2. **grid 用 lambda 接 META**：`lambda META: (triton.cdiv(nq, META['Q_TILE_SIZE']), bs)`。因为 tile size 现在由 autotune 决定，host 侧不知道选哪个，必须让 grid 依赖 autotune 选中的 config。
3. **三个 kernel 独立 autotune**。fwd 与 dq 按 Q 并行（Q_TILE 决定 program 数），dkdv 按 K 并行（K_TILE 决定 program 数）——最优 tile 组合天然不同（dkdv 里 K_TILE 更大能减少 program 数，Q_TILE 是内层可以更小）。fwd 和 dq 结构相同，直接共用一份 config 列表。
4. **手挑 pareto 前沿 8 组，不爆搜**：Q_TILE ∈ {32, 64, 128}，K_TILE ∈ {32, 64}（dkdv 加 128），num_warps ∈ {4, 8}，num_stages ∈ {2, 3}。全乘 = 24 组，首次编译 30~60 秒；挑 8 组既能覆盖 pareto 又控在 10 秒内。
5. **非法 config 自动淘汰**：某些组合（如 Q_TILE=128, K_TILE=64, num_warps=8, num_stages=3 在 d=128 bf16 上）可能寄存器爆或 shared memory 溢出，Triton 编译或运行时会 fail，autotune 自动跳过该 config——不需要手工排除。

**潜在坑（已验证 OK）**：
- 单元测试 `seq=128, d=64, fp32`：configs 里最大 Q_TILE=128 恰好整除，`boundary_check` 兜底越界 tile。若将来 test 用 seq<32 会出问题，届时补 `min(tile, seq_len)` 逻辑或加更小的 config。
- Leaderboard `seq=32768, d=128, bf16`：所有 config 都能塞下 shared memory（bf16 128×128 tile = 32KB，加 Q/K/V/O 与 fp32 accumulator 共 ~160KB < H100/H20 228KB/SM）。
- Autotune 缓存 key 命中后开销为 0；不同 (seq, d) 各自 sweep 一次。leaderboard 只有一种形状，10 分钟启动预算充裕。

**与前面各步的组合关系**：#3+#4+#5 都改 `opt_kernel.py`，是同一个 attention kernel 的三层优化——#3 少算，#4 少判断，#5 找最优参数。#5 是"参数调优"，前两个是"算法/代码调优"，正交且叠加。

---

## #6 已完成详情（Activation checkpointing）

- 实现文件：`cs336_systems/activation_ckpt.py`，导出 `CheckpointedBlock` + `apply_activation_checkpointing(model)`。**不改 `cs336-basics/model.py`**（那是共享代码），改用 nn.Module 包装每个 TransformerBlock 就地替换 `model.layers[i]`。
- 测试文件：`tests/test_activation_ckpt.py`。正确性用 2 层小模型对比 checkpoint 前后 loss + 所有参数梯度（fp32 逐位 1e-5、bf16 宽松 2e-2）；`main()` 用 8 层 d_model=1024 seq=2048 做显存对比。

**为什么这里靠激活重算最省显存**：8B 模型的显存开销分三块——静态（参/梯/优化器 ~65GB，#7 用 FSDP 切）+ logits/CE 中间量（40GB，#1 已解决）+ **每层激活**（250~320GB，本题靶子）。前向阶段每个 TransformerBlock 内部要保存下来给反向用的中间张量非常多：ln1/ln2 输出、q/k/v proj 输出（各 `[B,S,d_model]`）、attention 权重（Flash 里省掉但 output `[B,H,S,d_head]` 还要）、SwiGLU 里 `w1(x)`、`w3(x)`、`silu(w1(x))*w3(x)`（`[B,S,d_ff]` × 3）。粗算一层 bf16 ≈ 10GB，34 层就是 340GB——**不 checkpoint 根本塞不下 seq=32768**，B200 180GB 都不够。

**activation checkpointing 的机制**：
1. **前向**：对每个 block 只保留 **输入张量**（`[B,S,d_model]` bf16，@2×32k×4096 = 512MB），block 内所有中间激活算完立刻丢，autograd 图里对该 block 打一个"checkpoint 标记"。
2. **反向**：走到该 block 时，用保存的输入 **重跑一次前向**（在 `torch.enable_grad()` 下重建局部计算图），拿到中间激活后立刻做反向，backward 完释放。
3. **代价**：每层多一次 forward，训练总 FLOPs ≈ 1.33x（forward 33% + backward 67% → +33%），是算力换显存的经典做法。

**显存账**：34 层输入张量总驻留 34 × 512MB ≈ 17GB；反向时同一时刻只有 1 层的完整中间激活活着（~10GB）；峰值激活约 27GB（vs 原 340GB），加上参/梯/优化器和 fused CE 后完全塞得下 B200/H20。

**设计要点**：
1. **`use_reentrant=False`**（新 API）：不走 autograd re-entry，兼容任意输入/输出结构、自动保存 RNG state、和 `torch.compile` 良好协作。老的 `reentrant=True` 对 kwargs、non-tensor 输入敏感，且 compile 下容易出问题——直接用新 API。
2. **`CheckpointedBlock(nn.Module)` 包装而非 monkey-patch `forward`**：原 block 作为 `self.block` 子模块，`state_dict`、`named_parameters` 等自然工作；测试里两份模型的 `list(parameters())` 顺序完全一致，可直接 zip 对齐（只是名字前多个 `block.`，无所谓）。
3. **只包 `TransformerBlock`，不碰 embedding / final norm / lm_head**：Embedding 输入是 int64（`checkpoint` 对 non-float tensor 会警告且不需要），final norm 和 lm_head 单独重算意义不大且后接 fused CE 本身就省。
4. **无随机性副作用前提**：模型没启用 dropout，attention 走确定的 Triton kernel，所以 checkpoint 前后数值可逐位一致（fp32 测试就是这么验证的）。若将来加 dropout，`use_reentrant=False` 会自动 save/restore RNG state。

**待集成（#7）**：leaderboard 训练步在构造完模型后调 `apply_activation_checkpointing(model)`。与 FSDP 组合时要放在 FSDP 包装之**前**（FSDP 见到的应是已经 checkpoint 化的 block）。

---

## #7 已完成详情（端到端集成：FSDP + torch.compile + full-step 计时）

- 新增文件：`cs336_systems/leaderboard_integration.py`（集成层，**不改 basics/model.py**）+ 改写 `cs336_systems/leaderboard.py`（2 卡分布式 full-step 计时）+ `cs336_systems/dist_train.py` 的 `FSDP` 加 `skip_modules` 参数。
- 测试文件：`tests/test_leaderboard_integration.py`（单卡小模型冒烟）：① Flash patch 后整模型前向 ≈ naive；② forward_hidden+fused CE 的 loss/梯度 == 朴素 logits+CE（fp32 逐位）；③ Flash+ckpt+fused CE 全叠加跑通且梯度有限。

**集成层做的三件事（都靠"不动共享代码"的手法）**：
1. **Flash attention（#3/#4/#5）接入**：`patch_flash_attention()` monkey-patch 模块级 `cs336_basics.model.scaled_dot_product_attention → flash_sdpa`。模型 attention 层内部调的就是这个模块函数，patch 掉即全局生效。`flash_sdpa` 把模型的 4D `[B,H,S,d]` reshape 成 kernel 要的 3D `[B*H,S,d]`、`is_causal=True`（kernel 自己走下三角，比传 mask 更省）、结果 reshape 回去；传入的 causal mask 忽略。
2. **Fused CE + LM head（#1）接入**：`forward_hidden(model, x)` 只跑 embedding→layers→ln_final 返回**隐藏态**（不过 lm_head、不 materialize `[B,S,vocab]` 全量 logits）；loss 用 `fused_linear_cross_entropy(hidden, lm_head.weight, targets)` 分块算。注意 `nn_utils.cross_entropy` 与 fused CE **都是 mean-over-tokens**，直接对齐（原 leaderboard 的 `.sum()` 作用在标量上是恒等）。
3. **Activation ckpt（#6）**：`apply_activation_checkpointing(model)` 在 `.to(bf16).cuda()` 之后、FSDP 之前调。

**leaderboard.py 的组装顺序（关键，顺序错会出 bug）**：
```
patch_flash_attention()                       # ① 先 patch，之后建的模型 attention 全走 Flash
base = build_model().to(cuda, bf16)           # ② 建 8B 模型上卡
apply_activation_checkpointing(base)          # ③ 每层包 checkpoint（必须在 FSDP 之前）
model = HiddenModel(base)                      # ④ 包成"出隐藏态"的 nn.Module（overlap 关键，见坑 B）
fsdp = FSDP(model, prefetch=2,
            skip_modules={base.lm_head})       # ⑤ 切分静态状态 + prefetch overlap
optimizer = FusedAdamW(model.parameters())    # ⑥ #2 的优化器
# train_step: hidden = fsdp(labels) → fused_linear_cross_entropy(hidden, base.lm_head.weight, ...)
```

**#7 里踩到并解决的三个真实坑**：
1. **lm_head 不能被 FSDP 切分**。FSDP 靠 forward hook 在进入 Linear 时 all-gather 完整 weight、退出再切回。但 fused CE 里我**手动读 `lm_head.weight`**（`x@W.T`），绕过了 lm_head 的 forward hook → 若被切分只能拿到残缺分片。解决：给 `FSDP` 加 `skip_modules` 参数，把 `lm_head` 排除切分（622M 参数 bf16 每卡多留 ~1.2GB，可接受）。embedding 走正常 `token_embeddings(x)` 有 hook，照常切。
2. **prefetch 依赖前向走顶层 `FSDP.forward` → 用 `HiddenModel` wrapper 打开 overlap**。prefetch（异步 all-gather 后 N 层 weight 与本层计算 overlap）的 inflight 队列是在**顶层 `FSDP.forward`** 里初始化的。最初 `forward_hidden` 直接调子模块、绕过顶层 forward，所以只能退回 `prefetch=0`（同步 gather，无 overlap）。**方案 1（已落地）**：把 `forward_hidden` 包成 `HiddenModel(base)` 的 `forward`，用 `FSDP(HiddenModel(base))`，训练步 `hidden = fsdp(labels)` 就会先跑 `FSDP.forward`（初始化 prefetch）再进逐层前向 → 每层 `_pre_forward` 走 `prefetch>0` 分支，异步预取后 2 层 weight，与本层计算 overlap。lm_head 仍在 wrapper 外由 fused CE 手算、仍从 `skip_modules` 排除，#1 收益不变。
3. **组装顺序里 `HiddenModel` 包装在 activation ckpt 之后**：ckpt 作用在 `base`（含 `.layers`），wrapper 只是再套一层 forward，不影响 ckpt 结构；FSDP 遍历 `HiddenModel → base → 各 Linear/Embedding` 的深度优先顺序 = 定义顺序 = 执行顺序，保证 prefetch 预取的"下 N 层"确为执行上的下 N 层。

**activation ckpt × FSDP prefetch 的交互（已确认正确，但重算路径无 overlap）**：checkpoint 在反向阶段重算 block forward 时，会再次触发每层 `_pre_forward`。此时**不经过顶层 `FSDP.forward`**，`_inflight` 为空 → `_materialize` 命中 `else` 分支退回**同步 `_gather_full`**（拿到的仍是完整 weight，数值正确，只是这次重算没享受到 overlap）。`_prefetch_next` 在重算里预取的后续层 handle 会残留 `_inflight`，下一步 `train_step` 的 `fsdp(labels)` 开头 `_inflight.clear()` 清掉——功能正确。重算路径的 overlap 属更深优化（方案 3），暂不做。

**torch.compile 现状**：已从主路径移除（前向改走 `fsdp(...)` 顶层，compile 套 FSDP.forward 的 hook 副作用极易 graph break）。若要做图优化，正道是换官方 `fully_shard` + `torch.compile`，但本作业用自实现 FSDP，不引入。

**如何跑（H20/B200，2 卡，工作目录内`）**：
```bash
# 先跑单卡集成冒烟（快，验证 Flash+ckpt+fused CE 数值正确 + HiddenModel 等价）
uv run pytest tests/test_leaderboard_integration.py -v -s

# 端到端 full-step 计时（2 卡，8B，seq=32768）。首次含 Triton autotune + cuBLAS 选 kernel。
LB_WARMUP=3 LB_REP=5 uv run python -m cs336_systems.leaderboard
```
输出示例：`[rank0] full-step (fwd+loss+bwd+AdamW) = XXX ms/step | peak mem = YY.YY GB | prefetch=2`。
H20 绝对时间不可与 B200 直接比（算力 ~1/10），但可验证**端到端能在显存内跑通**、各优化协同无误、以及 prefetch overlap 相对 `prefetch=0` 的相对加速；B200 上的绝对 wall-clock 才是 leaderboard 打榜数字。

---

## 相关文件索引
- Flash 前反向 kernel：`cs336_systems/opt_kernel.py`（`flash_fwd_kernel` / `flash_bwd_dq_kernel` / `flash_bwd_dkdv_kernel`，#3/#4 改这里）。
- 模型：`cs336-basics/cs336_basics/model.py`（`BasicsTransformerLM`、`scaled_dot_product_attention`、lm_head）。
- CE 参考：`cs336-basics/cs336_basics/nn_utils.py`。
- 测试适配：`tests/adapters.py`。
- 分布式：`cs336_systems/dist_train.py`（`FSDP` 加了 `skip_modules`，#7 用）/ `benchmark_fsdp.py`。
- Activation ckpt（#6）：`cs336_systems/activation_ckpt.py`。
- 集成层（#7）：`cs336_systems/leaderboard_integration.py`（`flash_sdpa` / `patch_flash_attention` / `forward_hidden` / `HiddenModel`）。
- 端到端计时（#7）：`cs336_systems/leaderboard.py`。

## 结果情况
### #1：Fused CE + LM Head
[N=8192 D=1024 V=50000 torch.bfloat16]
  naive  peak mem = 3.433 GB
  fused  peak mem = 2.663 GB  (1.3x less)

### #2：Fused AdamW
[params=1.86B  torch.bfloat16]
  AdamW      step =   40.607 ms
  FusedAdamW step =    7.554 ms  (5.38x faster)

### #3：Flash causal early stop
| dtype | d | seq_len | Triton fwd (ms) | Triton bwd (ms) | Triton e2e (ms) | PyTorch fwd (ms) | PyTorch bwd (ms) | PyTorch e2e (ms) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 64 | 4096 | 0.068 | 0.192 | 0.257 | 0.281 | 0.246 | 0.524 |
| bf16 | 64 | 8192 | 0.148 | 0.488 | 0.651 | 1.108 | 0.821 | 1.929 |
| bf16 | 64 | 16384 | 0.459 | 1.193 | 1.643 | 3.371 | 3.059 | 6.426 |

### #4：Flash L/D tile 分离
|dtype | d | seq_len | Triton fwd (ms) | Triton bwd (ms) | Triton e2e (ms) | PyTorch fwd (ms) | PyTorch bwd (ms) | PyTorch e2e (ms) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 64 | 4096 | 0.067 | 0.177 | 0.249 | 0.283 | 0.247 | 0.526 |
| bf16 | 64 | 8192 | 0.162 | 0.447 | 0.608 | 1.110 | 0.824 | 1.931 |
| bf16 | 64 | 16384 | 0.382 | 1.098 | 1.474 | 3.372 | 3.060 | 6.433 |
| bf16 | 128 | 4096 | 0.103 | 0.330 | 0.432 | 0.317 | 0.320 | 0.634 |
| bf16 | 128 | 8192 | 0.280 | 0.736 | 1.012 | 1.238 | 1.088 | 2.327 |
| bf16 | 128 | 16384 | 0.755 | 2.452 | 3.199 | 3.877 | 4.093 | 7.971 |

### #5：Triton autotune（tile size / num_warps / num_stages）
| dtype | d | seq_len | Triton fwd (ms) | Triton bwd (ms) | Triton e2e (ms) | PyTorch fwd (ms) | PyTorch bwd (ms) | PyTorch e2e (ms) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 64 | 16384 | 0.385 | 1.133 | 1.509 | 3.372 | 3.055 | 6.432 |
| bf16 | 64 | 32768 | 1.249 | 4.016 | 5.252 | 13.057 | 12.132 | 25.210 |
| bf16 | 128 | 16384 | 0.733 | 2.224 | 2.947 | 3.870 | 4.075 | 7.940 |
| bf16 | 128 | 32768 | 2.434 | 7.985 | 10.434 | 15.176 | 16.478 | 31.707 |

### #6：Activation checkpointing
[num_layers=8 d_model=1024 seq=2048 bs=2 torch.bfloat16]
  no ckpt   peak mem = 7.444 GB
  activation-ckpt peak mem = 2.232 GB  (3.34x less)

### #7：端到端集成（2 卡 FSDP + Flash + ckpt + fused CE/AdamW）
[8B, batch=2, seq=32768, bf16, 2×H20, prefetch=2, warmup=3 rep=5]
  full-step (fwd+loss+bwd+AdamW) = 28037.0 ms/step
  peak mem/rank = 61.78 GB  （H20 98GB，显存非常宽裕，2 卡足够，无需上 4 卡）

说明：
- 28s/step 是 2×H20 数字，**不可与官方 2×B200 的 10s 基线直接比**（H20 单卡算力约 B200 ~1/10）。
  本结果证明的是"端到端能在显存内跑通 + 7 项优化协同正确"，B200 上的绝对 wall-clock 才是打榜数字。
- peak 61.78GB/卡 远低于 98GB：静态状态已被 FSDP 切到每卡 ~1/2，激活被 #6 ckpt 压到只剩重算峰值，
  fused CE 干掉了 40GB logits——三项省显存优化叠加的直接体现。
- 运行时的 `UserWarning: Full backward hook is firing ...` 可忽略：这是 FSDP 的 `_pre_backward`
  (register_full_backward_pre_hook) 在 activation-ckpt 的 detached 输入下触发的提示；我们的 hook 只借
  该时机 gather weight、不依赖 grad_output，逻辑正确（冒烟测试梯度逐位对齐已实证）。

