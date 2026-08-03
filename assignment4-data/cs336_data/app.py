from fastwarc.warc import ArchiveIterator, WarcRecordType
from pathlib import Path
from cs336_data.extract import extract_text_from_html_bytes
import random
from cs336_data.language_id import identify_language
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.harmful_detect import nsfw_detect, toxic_detect
from cs336_data.quality import gopher_quality_filter
random.seed(336)

def read_warc():
    path = Path(__file__).parent.parent / "data/CC/example.warc.gz"
    for record in ArchiveIterator(open(path, "rb")):
        if record.headers['WARC-Type'] == 'response':
            url = record.headers['WARC-Target-URI']
            html_bytes = record.reader.read()
            my_text = extract_text_from_html_bytes(html_bytes)
            yield url, my_text
    
def extract_warc_read():
    for i, (url, text) in enumerate(read_warc()):
        print(url)
        print(text)      
        print("=" * 40)
        if i >= 2:             
            break

def warc_id():
    results = []
    for url, text in read_warc():
        if not text.strip():      # 跳过空文本
            continue
        lang, score = identify_language(text)
        results.append((url, text, lang, score))
    n = len(results)
    n_en = sum(1 for _, _, lang, _ in results if lang == "en")
    print(f"Total: {n}, English: {n_en}, Percentage: {n_en / n:.2%}")
    scores = sorted(s for _, _, _, s in results)
    print(f"Min score: {scores[0]}, Max score: {scores[-1]}, median score: {scores[n // 2]}")
    for url, text, lang, score in random.sample(results, min(20, n)):
        print("=" * 40)
        print(url)
        print(text[:300])
        print(f"Language: {lang}, Score: {score}")

def warc_pii():
    results = []
    for url, text in read_warc():
        if not text.strip():      # 跳过空文本
            continue
        text, num_masked1 = mask_emails(text)
        text, num_masked2 = mask_phone_numbers(text)
        text, num_masked3 = mask_ips(text)
        results.append((url, text, num_masked1, num_masked2, num_masked3))
    n = len(results)
    for url, text, num_masked1, num_masked2, num_masked3 in random.sample(results, min(20, n)):
        print("=" * 40)
        print(url)
        print(text)
        print(f"Masked emails: {num_masked1}, Masked phone numbers: {num_masked2}, Masked IPs: {num_masked3}")

def warc_harmful():
    results = []
    n_nsfw = 0
    n_toxic = 0
    for url, text in read_warc():
        if not text.strip():          # 跳过空文本
            continue
        
        nsfw_label, nsfw_score = nsfw_detect(text)
        toxic_label, toxic_score = toxic_detect(text)

        is_nsfw = nsfw_label == "nsfw"
        is_toxic = toxic_label == "toxic"
        if is_nsfw:
            n_nsfw += 1
        if is_toxic:
            n_toxic += 1

        results.append((url, text, nsfw_label, nsfw_score, toxic_label, toxic_score))

    n = len(results)
    if n == 0:
        print("No documents.")
        return

    n_harmful = sum(
        1 for _, _, nl, _, tl, _ in results if nl == "nsfw" or tl == "toxic"
    )
    print(f"Total: {n}, NSFW: {n_nsfw}, Toxic: {n_toxic}, Harmful: {n_harmful} ({n_harmful / n:.2%})")

    for url, text, nsfw_label, nsfw_score, toxic_label, toxic_score in random.sample(results, min(20, n)):
        print("=" * 40)
        print(url)
        print(text)
        print(f"NSFW: {nsfw_label} ({nsfw_score:.3f}), Toxic: {toxic_label} ({toxic_score:.3f})")

def warc_quality():
    results = []
    for url, text in read_warc():
        if not text.strip():      # 跳过空文本
            continue
        result = gopher_quality_filter(text)
        results.append((url, text, result))
    n = len(results)
    for url, text, result in random.sample(results, min(20, n)):
        print("=" * 40)
        print(url)
        print(text[:300])
        print(f"Result: {result}")


if __name__ == "__main__":
    # extract_warc_read()
    # warc_id()
    # warc_pii()
    # warc_harmful()
    warc_quality()
        

