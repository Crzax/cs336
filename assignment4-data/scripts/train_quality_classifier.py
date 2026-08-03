"""训练 fastText 质量分类器。

用法:
    uv run python scripts/train_quality_classifier.py --epoch 25 --lr 0.2
"""

import argparse
from pathlib import Path

import fasttext

DATA_DIR = Path(__file__).parent.parent / "data/quality_classifier"
MODEL_OUT = Path(__file__).parent.parent / "data/classifiers/quality.bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    args = parser.parse_args()

    train_path = DATA_DIR / "train.txt"
    valid_path = DATA_DIR / "valid.txt"
    if not train_path.exists():
        raise SystemExit("缺少 train.txt，请先跑 build_quality_dataset.py")

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        dim=args.dim,
    )

    if valid_path.exists():
        n, precision, recall = model.test(str(valid_path))
        print(f"[valid] n={n} precision={precision:.4f} recall={recall:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"[save] {MODEL_OUT}")


if __name__ == "__main__":
    main()
