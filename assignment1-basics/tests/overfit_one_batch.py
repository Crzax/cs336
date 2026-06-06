# tests/overfit_one_batch.py
import torch
import numpy as np
from cs336_basics.nn.nn_transformer import TransformerLM
from cs336_basics.nn.nn_basic import cross_entropy
from cs336_basics.opt import AdamW

torch.manual_seed(0)
device = 'cuda:0'
B, L, V = 8, 64, 1000
model = TransformerLM(d_model=128, num_layers=2, num_heads=4, d_ff=256,
                      vocab_size=V, max_seq_len=L, theta=10000.0,
                      device=device, dtype=torch.float32)
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0)

# 一个固定的 batch
x = torch.randint(0, V, (B, L), device=device)
y = torch.randint(0, V, (B, L), device=device)

for step in range(500):
    logits = model(x)
    loss = cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"{step:4d}  loss {loss.item():.4f}")
