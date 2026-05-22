import json
from typing import Iterable, Iterator, BinaryIO
import re
import regex
import unicodedata

class tokenizer:
    def __init__(   
        self,  
        vocab:dict[int, bytes], 
        merges:list[tuple[bytes, bytes]],
        special_tokens:list[str] | None = None
    ):
        
        self.vocabi2b = vocab
        self.vocabb2i = {v : k for k, v in self.vocabi2b.items()}
        self.merges = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, 
        vocab_filepath:str, 
        merges_filepath:str, 
        special_tokens:list[str] | None = None
        ):
        def _get_gpt2_map():
            n = 0
            map = {}
            for i in range(256):
                if unicodedata.category(chr(i))[0] in ('L', 'N', 'P', 'S'):
                    map[i] = bytes([i])
                else:
                    map[n + 256] = bytes([i])
                    n += 1
            return map
        gpt2_map = _get_gpt2_map()

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_ori = json.load(f)
            vocab = {}
            for k, v in vocab_ori.items():
                vocab[v] = bytes(b"".join(gpt2_map[ord(c)] for c in k))

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = f.readlines()
            merges = [tuple(b"".join(gpt2_map[ord(c)] for c in k) for k in line.strip().split(" ")) for line in merges]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 1) 按 special_tokens 切分；special_tokens 为 None / [] 时，整段都按普通文本处理
        if self.special_tokens:
            # 长的优先，避免 "<|endoftext|><|endoftext|>" 被短的先吃掉
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            # 用捕获组，让 special token 自己留在 split 结果中
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
            parts = re.split(pattern, text)
            specials_set = set(self.special_tokens)
        else:
            parts = [text]
            specials_set = set()

        # 2) 预分词：special token 单独成一个 "tuple"，其余走 PAT 拆字节
        pretoken_lst = []
        for part in parts:
            if not part:
                continue
            if part in specials_set:
                # 特殊 token 整体作为一个 bytes，跳过 BPE merge
                pretoken_lst.append((part.encode("utf-8"),))
            else:
                for match in regex.finditer(PAT, part):
                    token_bytes = match.group().encode("utf-8")
                    pretoken_lst.append(tuple(token_bytes[i:i+1] for i in range(len(token_bytes))))

        # 3) BPE merge：特殊 token 已经是单元素 tuple，不会被拆/合
        for i in range(len(pretoken_lst)):
            tokens = list(pretoken_lst[i])
            while len(tokens) >= 2:
                best_rank = None
                best_j = -1
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_j = j
                if best_j == -1:
                    break
                tokens = tokens[:best_j] + [tokens[best_j] + tokens[best_j+1]] + tokens[best_j+2:]
            pretoken_lst[i] = tuple(tokens)
        
        # 4) 查表得 id
        ids = []
        for tokens in pretoken_lst:
            for token in tokens:
                if token in self.vocabb2i:
                    ids.append(self.vocabb2i[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        allbytes = b"".join(self.vocabi2b[id] for id in ids)
        return allbytes.decode("utf-8", errors="replace")



       
    