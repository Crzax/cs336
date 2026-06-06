"""
Decoding / generation script.

Example:
    python -m cs336_basics.infer \
        --ckpt runs/exp1/final.pt \
        --vocab vocab.json --merges merges.txt \
        --prompt "Once upon a time" \
        --max_new_tokens 200 --temperature 0.8 --top_p 0.9 \
        --device cuda:0
"""
import argparse
import torch

from cs336_basics.nn.nn_transformer import TransformerLM
from cs336_basics.tokenizer import tokenizer as Tokenizer
from cs336_basics.nn.nn_basic import softmax


# ---------- 1. 工具：温度 + top-p 过滤 ----------
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
        return out
    return logits / temperature


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus sampling: 保留累积概率 >= top_p 的最小集合，其余 logit 设成 -inf。
    logits 形状 (..., V)。
    """
    if top_p is None or top_p >= 1.0 or top_p <= 0:
        return logits

    # 1) 排序并算累积概率
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = softmax(sorted_logits, dim=-1)
    cum_probs = sorted_probs.cumsum(dim=-1)

    # 2) 找出"被剔除的位置"——累积超过 top_p 的部分
    # 注意：要保留第一个超过阈值的那个 token（论文定义的最小集合），
    # 所以把 mask 整体右移一位。
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    # 3) 还原到原始顺序
    remove_unsorted = torch.zeros_like(remove)
    remove_unsorted.scatter_(dim=-1, index=sorted_idx, src=remove)

    return logits.masked_fill(remove_unsorted, float("-inf"))


# ---------- 2. 主生成函数 ----------
@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_id: int | None = None,
    max_seq_len: int = 1024,
    device: torch.device | str = "cpu",
) -> list[int]:
    """
    自回归生成 token id 序列。返回 prompt + 生成内容的完整 id list。
    """
    model.eval()
    ids = list(prompt_ids)
    # 一次性 batch=1 的 tensor，每步 cat 上新 token
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # 上下文超过模型最大长度时截断（最简单的滑窗，留住最近的）
        x_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]

        # 1) 前向，只取最后一个位置的分布
        logits = model(x_cond)               # (1, T, V)
        logits = logits[:, -1, :]            # (1, V)

        # 2) 温度缩放
        logits = apply_temperature(logits, temperature)

        # 3) top-p 截断
        logits = top_p_filter(logits, top_p)

        # 4) 采样
        probs = softmax(logits, dim=-1)    # (1, V)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 5) 拼接并检查终止条件
        x = torch.cat([x, next_id], dim=1)
        nid = next_id.item()
        ids.append(nid)
        if eot_id is not None and nid == eot_id:
            break

    return ids


# ---------- 3. CLI ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   type=str, required=True)
    p.add_argument("--vocab",  type=str, required=True, help="vocab.json")
    p.add_argument("--merges", type=str, required=True, help="merges.txt")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=1.0,
                   help="0 = greedy / argmax; 1 = no scaling; >1 = more random")
    p.add_argument("--top_p",          type=float, default=1.0,
                   help="nucleus sampling threshold; 1.0 = disabled")
    # 模型超参（必须和训练时一致）
    p.add_argument("--vocab_size",     type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--d_ff",           type=int, default=None)
    p.add_argument("--rope_theta",     type=float, default=10000.0)
    p.add_argument("--eot_token",      type=str, default="<|endoftext|>")
    p.add_argument("--device",         type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int, default=None)
    return p.parse_args()


def main():
    args = get_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1) tokenizer
    tok = Tokenizer.from_files(args.vocab, args.merges,
                               special_tokens=[args.eot_token])
    eot_id = tok.vocabb2i.get(args.eot_token.encode("utf-8"))

    # 2) model + 加载 checkpoint
    model = TransformerLM(
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
        d_ff=args.d_ff, vocab_size=args.vocab_size,
        max_seq_len=args.context_length, theta=args.rope_theta,
        device=device, dtype=torch.float32,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[load] {args.ckpt}  step={ckpt.get('iteration', '?')}")

    # 3) 编码 prompt
    prompt_ids = tok.encode(args.prompt)
    print(f"[prompt] {len(prompt_ids)} tokens: {args.prompt!r}")

    # 4) 生成
    out_ids = generate(
        model, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eot_id=eot_id,
        max_seq_len=args.context_length,
        device=device,
    )

    # 5) 解码并打印
    text = tok.decode(out_ids)
    print("\n========== generation ==========")
    print(text)
    print("================================")
    print(f"[stats] generated {len(out_ids) - len(prompt_ids)} new tokens "
          f"(stopped on EOT: {out_ids[-1] == eot_id if eot_id is not None else False})")


if __name__ == "__main__":
    main()
