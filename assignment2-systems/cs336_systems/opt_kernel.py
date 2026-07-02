import torch
from einops import einsum
import triton
import triton.language as tl

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
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        Sij = tl.dot(Qi, Kj.T, Sij) * scale
        if is_causal:
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
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    d_out_block_ptr = tl.make_block_ptr(
        d_out_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
        d_out = tl.load(d_out_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, K.T) * scale
        if is_causal:
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
        dVj = tl.dot(Pij.T, d_out, dVj)
        dKj = tl.dot(dSij.T, Qi, dKj)
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
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')
        Sij = tl.dot(Qi, Kj.T) * scale
        if is_causal:
            q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]      # [bq, bk]，允许 q>=k
            Sij = tl.where(mask, Sij, -1e6)
        Pij = tl.exp(Sij - Li[:, None])
        dPij = tl.dot(d_out, Vj.T)
        dSij = Pij * (dPij - Di[:, None])
        dQij = tl.dot(dSij, Kj, dQij)
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
        L = torch.empty((bs, nq), device=Q.device, dtype=Q.dtype)
        q_tile_size = 16
        k_tile_size = 16
        
        flash_fwd_kernel[triton.cdiv(nq, q_tile_size), bs] (
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            nq, nk,
            1.0 / d**0.5,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
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
        q_tile_size = 16
        k_tile_size = 16
        flash_bwd_dkdv_kernel[(triton.cdiv(nk, k_tile_size), bs)](
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
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=ctx.is_causal,
        ) 
        flash_bwd_dq_kernel[(triton.cdiv(nq, q_tile_size), bs)](
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
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=ctx.is_causal,
        )
        return dQ, dK, dV, None