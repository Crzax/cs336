import torch
import torch.nn as nn
from einops import einsum
from math import sqrt

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(
                            nn.init.trunc_normal_(
                                tensor = torch.empty((out_features, in_features), device=device, dtype=dtype),
                                mean = 0, std=std,
                                a = -3 * std,
                                b = 3 * std, 
                                )
                            )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        std= 1
        self.weight = nn.Parameter(
                            nn.init.trunc_normal_(
                                tensor = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                                mean = 0, std=std,
                                a = -3 * std,
                                b = 3 * std, 
                                )
                            )
        self.device = device
        self.dtype = dtype
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]