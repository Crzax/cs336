import torch
from einops import einsum
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs（#5）
# ---------------------------------------------------------------------------
# 三个 kernel 独立 autotune：fwd 与 dq 都按 Q 并行内循环 K，dkdv 反过来。
# 挑 pareto 前沿 6 组，避开 shared memory 溢出与低 SM 占用两个极端：
#   - Q_TILE / K_TILE ∈ {32, 64, 128}：太小 SM 利用不足；太大 shared memory 溢出。
#   - num_warps ∈ {4, 8}
#   - num_stages ∈ {2, 3}：Hopper 上 async pipeline 深度；反向 kernel 载入 5 个 tensor（K/V/Q/dO/O）
#     + fp32 accumulator，num_stages=3 会把 buffer 复制 3 份，shared memory 逼近 SM 上限 228KB，
#     大 tile 组合下容易 illegal memory access。反向一律用 num_stages=2 稳妥。
# key 是 (N_QUERIES, N_KEYS, D, is_causal)——决定最优 config 的所有维度，dtype 由 constexpr 自动特化。
_FWD_CONFIGS = [
    triton.Config({'Q_TILE_SIZE': 32,  'K_TILE_SIZE': 32}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 32}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 32}, num_warps=4, num_stages=3),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64}, num_warps=4, num_stages=3),
    triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=2),
]
# 反向 dq 与 fwd 结构相同（按 Q 并行），但反向内循环还要载入 Vj/dOi/Li，寄存器压力更大——
# 沿用同一批 config，但去掉最大 tile + 高 stages 的极端组合（fwd 里也已剔除）。
_BWD_DQ_CONFIGS = _FWD_CONFIGS
# 反向 dkdv 按 K 并行、外循环 Q，K_TILE 决定 program 数量（越大 program 越少），
# 但 K/V 是持久 tile（不换），Q/dO/O/L 是流式载入——K_TILE 太大反而寄存器紧张。
# 一律 num_stages=2 避免 buffer 3 份触发 shmem 越界（seq=32768 下亲测大 tile+3 stages 崩）。
_BWD_DKDV_CONFIGS = [
    triton.Config({'Q_TILE_SIZE': 32,  'K_TILE_SIZE': 32},  num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 32,  'K_TILE_SIZE': 64},  num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64},  num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64},  num_warps=8, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 32,  'K_TILE_SIZE': 128}, num_warps=8, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 128}, num_warps=8, num_stages=2),
]
_AUTOTUNE_KEY = ['N_QUERIES', 'N_KEYS', 'D', 'is_causal']


def _pick_tile(seq_len: int, d: int = 64, dtype=torch.float32) -> int:
    # head dim 越大，tile x d 占的共享内存越多；tl.dot 要求维度 >= 16，
    # tile 不超过序列长度。fp32 每元素 4B，d=128 时 tile=64 会超 shared memory，
    # 需收缩到 32；bf16 每元素 2B，占用减半，d=128 仍可用 tile=64。
    if seq_len >= 128:
        tile = 64
    elif seq_len >= 64:
        tile = 32
    else:
        tile = 16
    # 仅 fp32（4 字节）在 d>=128 时才需要压缩 tile 以避免共享内存溢出
    if d >= 128 and dtype == torch.float32:
        tile = min(tile, 32)
    return min(tile, seq_len)

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Nq, d = Q.shape[-2], Q.shape[-1]
        Nk, d = K.shape[-2], K.shape[-1]
        Bq, Bk = 16, 16
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        for i in range(Tq):
            Qi = Q[..., i*Bq:(i+1)*Bq, :]
            Oi = torch.zeros_like(Qi)
            li = torch.zeros(Qi.shape[:-1], device=Q.device, dtype=Q.dtype)
            mi = torch.full(Qi.shape[:-1], -torch.inf, device=Q.device, dtype=Q.dtype)
            for j in range(Tk):
                Kj = K[..., j*Bk:(j+1)*Bk, :]
                Vj = V[..., j*Bk:(j+1)*Bk, :]                
                Sij = einsum(Qi, Kj, '... Bq d, ... Bk d -> ... Bq Bk') / d**0.5
                if is_causal:
                    q_pos = torch.arange(i*Bq, i*Bq + Qi.shape[-2], device=Q.device)
                    k_pos = torch.arange(j*Bk, j*Bk + Kj.shape[-2], device=Q.device)
                    mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
                    Sij = Sij.masked_fill(~mask, float('-inf'))

                mij = torch.maximum(mi, torch.max(Sij, dim=-1).values)
                Pi = torch.exp(Sij - mij[..., None])
                lij = torch.exp(mi-mij)*li + torch.sum(Pi, dim=-1)
                Oi = torch.exp(mi-mij)[..., None]*Oi + Pi@Vj
                li = lij
                mi = mij
            Oi = (1.0 / li)[..., None] * Oi
            Li = mi + torch.log(li)
            O[..., i*Bq:(i+1)*Bq, :] = Oi
            L[..., i*Bq:(i+1)*Bq] = Li
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, d_out):
        def _backward():
            L, Q, K, V, O = ctx.saved_tensors
            d = Q.shape[-1]
            S = Q @ K.mT / d**0.5
            if ctx.is_causal:
                q_pos = torch.arange(S.shape[-2], device=Q.device)
                k_pos = torch.arange(S.shape[-1], device=Q.device)
                mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
                S = S.masked_fill(~mask, float('-inf'))
            P = torch.exp(S - L[..., None])
            dV = P.mT @ d_out
            dP = d_out @ V.mT
            Di = torch.sum(O * d_out, dim=-1)
            dS = P * (dP - Di[..., None])
            dQ = dS @ K / d**0.5
            dK = dS.mT @ Q / d**0.5
            return dQ, dK, dV, None
        return torch.compile(_backward)()
        

@triton.autotune(configs=_FWD_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE, ), float('-inf'), dtype=tl.float32)
    K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
    V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
    # causal 下 K tile 分三类（#3 跳纯 0 tile + #4 L/D 分离）：
    #   1. 纯下三角块（K tile 最大 key <= Q tile 最小 query）：结果全合法，无需 mask。
    #   2. 对角块（Q/K tile 骑跨对角线）：需要逐元素 q>=k 比较 + where。
    #   3. 纯上三角块（K tile 最小 key > Q tile 最大 query）：全 0，直接不迭代（#3）。
    # 前向按 query tile 并行，本 Q tile 最小 query = qt*Q_TILE，最大 query = (qt+1)*Q_TILE-1。
    #   - 纯下三角块条件：(kt+1)*K_TILE - 1 <= qt*Q_TILE，即 kt < qt*Q_TILE/K_TILE（下取整）。
    #     故非对角块个数 = qt*Q_TILE // K_TILE。
    #   - 对角块最多需要迭代到覆盖 (qt+1)*Q_TILE-1 的 K tile，即 cdiv((qt+1)*Q_TILE, K_TILE)。
    # 分成两段循环：非对角段完全无 mask（编译器可生成纯 GEMM 无分支）；对角段仅少量 tile 走 mask。
    if is_causal:
        n_full_tiles = (query_tile_index * Q_TILE_SIZE) // K_TILE_SIZE
        n_total_tiles = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    else:
        n_full_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        n_total_tiles = n_full_tiles
    # 段 1：非对角块（无 mask，纯 GEMM）
    for key_tile_index in range(n_full_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        Sij = tl.dot(Qi, Kj.T, Sij) * scale
        mij = tl.maximum(mi, tl.max(Sij, axis=1))
        Pi = tl.exp(Sij - mij[:, None])
        lij = tl.exp(mi - mij) * li + tl.sum(Pi, axis=1)
        Oi = tl.exp(mi - mij)[:, None] * Oi
        Pi = Pi.to(Vj.dtype)
        Oi = tl.dot(Pi, Vj, Oi)
        li = lij
        mi = mij
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    # 段 2：对角块（有 mask）；非 causal 时 n_total_tiles == n_full_tiles，循环空转
    for key_tile_index in range(n_full_tiles, n_total_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        Sij = tl.dot(Qi, Kj.T, Sij) * scale
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
        k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
        mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
        Sij = tl.where(mask, Sij, -1e6)
        mij = tl.maximum(mi, tl.max(Sij, axis=1))
        Pi = tl.exp(Sij - mij[:, None])
        lij = tl.exp(mi - mij) * li + tl.sum(Pi, axis=1)
        Oi = tl.exp(mi - mij)[:, None] * Oi
        Pi = Pi.to(Vj.dtype)
        Oi = tl.dot(Pi, Vj, Oi)
        li = lij
        mi = mij
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = (1.0 / li)[:, None] * Oi
    Li = mi + tl.log(li)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    Oi = Oi.to(O_block_ptr.type.element_ty)
    Li = Li.to(L_block_ptr.type.element_ty)
    
    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

@triton.autotune(configs=_BWD_DKDV_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def flash_bwd_dkdv_kernel(
    L_ptr, Q_ptr, K_ptr, V_ptr, O_ptr, d_out_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_lb, stride_lq,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
    V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
    K = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
    V = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
    # causal early stop（此 kernel 按 key tile 并行，反过来跳 query tile）：
    # 本 K tile 最小 key 下标 = key_tile_index*K_TILE_SIZE；所有 max_q 小于它的
    # query tile 全 q<k 被 mask，无贡献 → 从含该 key 下标的 query tile 起算。
    # 起始 query tile = (key_tile_index*K_TILE_SIZE)//Q_TILE_SIZE，并把各 block_ptr
    # 的初始 offset 直接落到该 tile，省掉前面所有全 0 的迭代。
    if is_causal:
        start_query_tile = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
    else:
        start_query_tile = 0
    q_offset = start_query_tile * Q_TILE_SIZE
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    d_out_block_ptr = tl.make_block_ptr(
        d_out_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(q_offset, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    # dKdV kernel 按 key tile 并行，Q tile 分三类（反向对称于前向）：
    #   1. 纯上三角（qt*Q_TILE + Q_TILE - 1 < kt*K_TILE，即 K 在 Q 完全右侧）：全 0，直接不迭代（#3 用 start_query_tile 跳过）。
    #   2. 对角块（Q/K tile 骑跨对角线）：需要逐元素 mask。
    #   3. 纯下三角块（qt*Q_TILE >= (kt+1)*K_TILE，即 K 最大 key 严格小于 Q 最小 query）：全保留，无 mask。
    # 对角块 qt 范围：[start_query_tile, first_full_qt)，其中 first_full_qt = cdiv((kt+1)*K_TILE, Q_TILE)。
    # 非对角块 qt 范围：[first_full_qt, N_Q_tiles)。
    n_total_qt = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    if is_causal:
        first_full_qt = tl.cdiv((key_tile_index + 1) * K_TILE_SIZE, Q_TILE_SIZE)
    else:
        first_full_qt = start_query_tile  # 非 causal 时全部按无 mask 走
    # 段 1：对角块（有 mask）；非 causal 时该循环空转
    for query_tile_index in range(start_query_tile, first_full_qt):
        Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
        d_out = tl.load(d_out_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, K.T) * scale
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
        k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
        mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
        Sij = tl.where(mask, Sij, -1e6)
        Li = tl.load(L_block_ptr, boundary_check=(0,))
        Pij = tl.exp(Sij - Li[:, None])
        O = tl.load(O_block_ptr, boundary_check=(0,1), padding_option='zero')
        Di = tl.sum(d_out * O, axis=1)
        dPij = tl.dot(d_out, V.T)
        dSij = Pij * (dPij - Di[:, None])
        dVj = tl.dot(Pij.T.to(d_out.dtype), d_out, dVj)
        dKj = tl.dot(dSij.T.to(Qi.dtype), Qi, dKj)
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        d_out_block_ptr = tl.advance(d_out_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
    # 段 2：非对角块（无 mask，纯 GEMM）
    for query_tile_index in range(first_full_qt, n_total_qt):
        Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
        d_out = tl.load(d_out_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, K.T) * scale
        Li = tl.load(L_block_ptr, boundary_check=(0,))
        Pij = tl.exp(Sij - Li[:, None])
        O = tl.load(O_block_ptr, boundary_check=(0,1), padding_option='zero')
        Di = tl.sum(d_out * O, axis=1)
        dPij = tl.dot(d_out, V.T)
        dSij = Pij * (dPij - Di[:, None])
        dVj = tl.dot(Pij.T.to(d_out.dtype), d_out, dVj)
        dKj = tl.dot(dSij.T.to(Qi.dtype), Qi, dKj)
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        d_out_block_ptr = tl.advance(d_out_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
    
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dKj = (dKj * scale).to(dK_block_ptr.type.element_ty)
    dVj = dVj.to(dV_block_ptr.type.element_ty)
    tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))

@triton.autotune(configs=_BWD_DQ_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def flash_bwd_dq_kernel(
    L_ptr, Q_ptr, K_ptr, V_ptr, O_ptr, d_out_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_lb, stride_lq,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O = tl.load(O_block_ptr, boundary_check=(0,1), padding_option='zero')
    d_out_block_ptr = tl.make_block_ptr(
        d_out_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    d_out = tl.load(d_out_block_ptr, boundary_check=(0,1), padding_option='zero')
    Di = tl.sum(O * d_out, axis=-1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
    Li = tl.load(L_block_ptr, boundary_check=(0,))
    dQij = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    # causal L/D 分离：非对角块无 mask（纯 GEMM），对角块少量 tile 走 mask。切分同前向。
    if is_causal:
        n_full_tiles = (query_tile_index * Q_TILE_SIZE) // K_TILE_SIZE
        n_total_tiles = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    else:
        n_full_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        n_total_tiles = n_full_tiles
    # 段 1：非对角块（无 mask）
    for key_tile_index in range(n_full_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, Kj.T) * scale
        Pij = tl.exp(Sij - Li[:, None])
        dPij = tl.dot(d_out, Vj.T)
        dSij = Pij * (dPij - Di[:, None])
        dQij = tl.dot(dSij.to(Kj.dtype), Kj, dQij)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    # 段 2：对角块（有 mask）
    for key_tile_index in range(n_full_tiles, n_total_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, Kj.T) * scale
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
        k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
        mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
        Sij = tl.where(mask, Sij, -1e6)
        Pij = tl.exp(Sij - Li[:, None])
        dPij = tl.dot(d_out, Vj.T)
        dSij = Pij * (dPij - Di[:, None])
        dQij = tl.dot(dSij.to(Kj.dtype), Kj, dQij)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQij = (dQij * scale).to(dQ_block_ptr.type.element_ty)
    tl.store(dQ_block_ptr, dQij, boundary_check=(0, 1))

        
class FlashAttnFuncTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        bs, nq, d = Q.shape
        bs, nk, d = K.shape
        O = torch.empty_like(Q)
        # L (log-sum-exp) 数值敏感，始终用 fp32 存储，与 dtype 无关
        L = torch.empty((bs, nq), device=Q.device, dtype=torch.float32)

        # grid 用 lambda 接受 META（autotune 选中的 config），保证 program 数量匹配 Q_TILE_SIZE。
        fwd_grid = lambda META: (triton.cdiv(nq, META['Q_TILE_SIZE']), bs)
        flash_fwd_kernel[fwd_grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            nq, nk,
            1.0 / d**0.5,
            D=d,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, d_out):
        L, Q, K, V, O = ctx.saved_tensors
        bs, nq, d = Q.shape
        bs, nk, d = K.shape
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        dkdv_grid = lambda META: (triton.cdiv(nk, META['K_TILE_SIZE']), bs)
        flash_bwd_dkdv_kernel[dkdv_grid](
            L, Q, K, V, O, d_out,
            dQ, dK, dV,
            L.stride(0), L.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nq, nk,
            1.0 / d**0.5,
            D=d,
            is_causal=ctx.is_causal,
        )
        dq_grid = lambda META: (triton.cdiv(nq, META['Q_TILE_SIZE']), bs)
        flash_bwd_dq_kernel[dq_grid](
            L, Q, K, V, O, d_out,
            dQ, dK, dV,
            L.stride(0), L.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nq, nk,
            1.0 / d**0.5,
            D=d,
            is_causal=ctx.is_causal,
        )
        return dQ, dK, dV, None