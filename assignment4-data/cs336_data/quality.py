import fasttext
import pathlib

def gopher_quality_filter(text: str) -> bool:
    words = text.split() 
    if len(words) < 50 or len(words) > 100000:
        return False
    alp_word = 0
    total_word_len = 0
    for word in words:
        total_word_len += len(word)
        if any(c.isalpha() for c in word):
            alp_word += 1
    if total_word_len / len(words) < 3 or total_word_len / len(words) > 10:
        return False
    if alp_word / len(words) < 0.8:
        return False
    
    lines = [ l for l in text.splitlines() if l.strip() ]
    line_count = len(lines)
    if line_count == 0:
        return False
    dots_end_count = 0
    for line in lines:
        if line.rstrip().endswith(("...",)):
            dots_end_count += 1
    if dots_end_count / line_count > 0.3:
        return False

    return True

MODEL = None

def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    model_path = pathlib.Path(__file__).parent.parent / "data/classifiers/quality.bin"
    MODEL = fasttext.load_model(str(model_path))
    return MODEL

def quality_classifier(text: str) -> tuple[str, float]:
    model = get_model()
    cleaned = text.replace("\n", " ").strip()
    labels, probs = model.predict(cleaned, k=1)
    label = labels[0].replace("__label__", "")    
    return label, float(probs[0])


