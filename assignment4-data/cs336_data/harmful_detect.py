import fasttext
import pathlib

NSFW_MODEL = None
TOXIC_MODEL = None

def nsfw_get_model():
    global NSFW_MODEL
    if NSFW_MODEL is not None:
        return NSFW_MODEL
    model_path = pathlib.Path(__file__).parent.parent / "data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
    NSFW_MODEL = fasttext.load_model(str(model_path))
    return NSFW_MODEL

def toxic_get_model():
    global TOXIC_MODEL
    if TOXIC_MODEL is not None:
        return TOXIC_MODEL
    model_path = pathlib.Path(__file__).parent.parent / "data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    TOXIC_MODEL = fasttext.load_model(str(model_path))
    return TOXIC_MODEL

def nsfw_detect(text: str) -> tuple[str, float]:
    model = nsfw_get_model()
    cleaned = text.replace("\n", " ").strip()
    labels, probs = model.predict(cleaned, k=1)
    label = labels[0].replace("__label__", "")    
    return label, float(probs[0])

def toxic_detect(text: str) -> tuple[str, float]:
    model = toxic_get_model()
    cleaned = text.replace("\n", " ").strip()
    labels, probs = model.predict(cleaned, k=1)
    label = labels[0].replace("__label__", "")    
    return label, float(probs[0])

