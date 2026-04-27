# profile_bpe.py
import cProfile
import pstats

from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(
        input_path="tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    profiler.disable()
    profiler.dump_stats("bpe.prof")  # 导出 .prof 文件，供 snakeviz 使用
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(30)  # top 30
