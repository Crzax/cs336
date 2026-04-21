import re
import regex

def train_bpe(input_path, vocab_size, special_tokens=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    @todo chunk
    """
    vocab = {i : bytes([i]) for i in range(256)} # 词表
    merges = [] # 合并结果

    with open(input_path, "rb") as f:
        # 分割语料
        data = f.read()
        text = data.decode("utf-8")
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, text)
        # 预分词
        # dict[tuple(bytes,...), int]
        word_counts = {} 
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:
            matches = regex.finditer(PAT, part)
            for match in matches:
                token_str = match.group()
                token_bytes = token_str.encode("utf-8")
                key = tuple(bytes([b]) for b in token_bytes)
                word_counts[key] = word_counts.get(key, 0) + 1
        
        pair_counts = {}
        pair_to_word = {}
        for word, count in word_counts.items():
            if len(word) > 1:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])        
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
                    pair_to_word.setdefault(pair, set()).add(word)

        nums_merge = vocab_size - 256 - len(special_tokens)
        for _ in range(nums_merge):
            max_pair= max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            pair_bytes = max_pair[0] + max_pair[1]
            merges.append(max_pair)
            vocab[len(vocab)] = pair_bytes
            
            # 合并 & update
            for word in list(pair_to_word[max_pair]):
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                            new_word.append(word[i] + word[i+1])
                            i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pair_counts[pair] -= word_counts[word]
                    pair_to_word[pair].discard(word)   
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                new_word = tuple(new_word)
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + word_counts[word]
                    pair_to_word.setdefault(pair, set()).add(new_word)
                word_counts[new_word] = word_counts.pop(word) + word_counts.get(new_word, 0)
            
            pair_counts.pop(max_pair, None)
            pair_to_word.pop(max_pair, None)
            
    for spt in special_tokens:
        vocab[len(vocab)] = spt.encode("utf-8")
    return (vocab, merges)