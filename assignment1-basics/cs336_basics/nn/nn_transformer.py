from this import d
import torch
import torch.nn as nn
from .nn_basic import *
from einops import einsum, rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x * self.weight / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None) -> None:
        super().__init__()
        if d_ff is None:
            multiple = 64
            d_ff = (round(8 / 3 * d_model) + multiple - 1) // multiple * multiple
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        return self.w2(x1 * torch.sigmoid(x1) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) :
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        position = torch.arange(max_seq_len, device=device)
        angles = einsum(position, freqs, 'seq_len, half_d -> seq_len half_d')
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x = rearrange(x, '... seq_len (d_k k) -> ... seq_len d_k k', k=2)
        out = torch.stack([
            x[..., 0] * cos - x[..., 1] * sin,
            x[..., 1] * cos + x[..., 0] * sin,
        ], dim = -1)
        return rearrange(out, '... seq_len d_k k -> ... seq_len (d_k k)')

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.amax(dim=dim, keepdim=True)
    return torch.exp(x - x_max) / torch.exp(x - x_max).sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(q, k ,v, mask=None):
    atten = einsum(q, k, 'batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k')
    atten = atten / q.size(-1) ** 0.5
    if mask is not None:
        atten = atten.masked_fill(mask == 0, float('-inf'))
    atten =  softmax(atten, dim=-1)
    return einsum(atten, v, 'batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None and max_seq_len is not None:
            self.rotary_positional_embedding = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x, tokens_positions=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = rearrange(q, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        
        seq_len = x.size(-2)
        assert tokens_positions is None or tokens_positions.size(-1) == seq_len
        if tokens_positions is not None:
            q = self.rotary_positional_embedding(q, tokens_positions)
            k = self.rotary_positional_embedding(k, tokens_positions)
        
        mask = torch.tril(torch.full((seq_len, seq_len), True, device=x.device))
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, "batch_size num_heads seq_len d_k -> batch_size seq_len (num_heads d_k)", num_heads=self.num_heads)
        return self.w_o(o)