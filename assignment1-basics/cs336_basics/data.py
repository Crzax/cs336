import numpy as np
import torch
import os
import typing

def data_loading(x:np.ndarray, batch_size: int, context_len: int, device: torch.device):
    N = x.size
    starts = np.random.randint(0, N - context_len, size=batch_size)
    idx = starts[:, None] + np.arange(context_len)[None, :]
    # uint16 不被 torch.from_numpy 支持；先 astype 成 int64
    # （顺便从 memmap 视图变成普通的 in-RAM 数组，断开 mmap 链接）
    now_token = np.asarray(x[idx], dtype=np.int64)
    nxt_token = np.asarray(x[idx + 1], dtype=np.int64)
    now_token = torch.from_numpy(now_token).to(device)
    nxt_token = torch.from_numpy(nxt_token).to(device)
    return now_token, nxt_token

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    itetration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': itetration
    }, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    ):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
    