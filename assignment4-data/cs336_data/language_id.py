import fasttext
import pathlib

_MODEL = None

# fastText lid.176 的标签基本已经是 ISO 639-1（en / zh ...），
# 这里保留一个映射表，方便以后遇到需要归一化的标签时改动。
_LABEL_REMAP = {
    "zh": "zh",
    "en": "en",
}


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model_path = pathlib.Path(__file__).parent.parent / "data/classifiers/lid.176.bin"
    _MODEL = fasttext.load_model(str(model_path))
    return _MODEL


def identify_language(text: str) -> tuple[str, float]:
    model = _get_model()

    # 1. 清洗：fastText 的 predict 不能吃换行符，否则会报错
    cleaned = text.replace("\n", " ").strip()

    # 2. 取 top-1 预测
    labels, probs = model.predict(cleaned, k=1)

    # 3. 从 '__label__xx' 里抠出语言码
    lang = labels[0].replace("__label__", "")

    # 4. remap（如有必要）
    lang = _LABEL_REMAP.get(lang, lang)

    # 5. 返回 (lang, float(score))
    return lang, float(probs[0])
