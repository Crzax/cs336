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
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None and max_seq_len is not None:
            self.rotary_positional_embedding = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x, tokens_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        
        seq_len = x.size(-2)
        assert tokens_positions is None or tokens_positions.size(-1) == seq_len

        if tokens_positions is None and hasattr(self, 'rotary_positional_embedding'):
            tokens_positions = torch.arange(seq_len, device=x.device)

        if tokens_positions is not None:
            q = self.rotary_positional_embedding(q, tokens_positions)
            k = self.rotary_positional_embedding(k, tokens_positions)
        
        mask = torch.tril(torch.full((seq_len, seq_len), True, device=x.device))
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, "batch_size num_heads seq_len d_k -> batch_size seq_len (num_heads d_k)", num_heads=self.num_heads)
        return self.output_proj(o)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x, tokens_positions=None):
        x = x + self.attn(self.ln1(x), tokens_positions)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids, tokens_positions=None):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, tokens_positions)
        x = self.ln_final(x)
        return self.lm_head(x)