"""Answer parsers for the safety/RLHF supplement evaluations."""

from __future__ import annotations

import re
from typing import Any

MMLU_LETTERS = ("A", "B", "C", "D")

# Ordered from most to least trustworthy. Every pattern captures a single letter
# and refuses to fire when the letter is glued to more letters (so the "B" in
# "BK virus" is never read as a prediction).
_MMLU_PATTERNS = (
    r"correct answer is[:\s]*\(?([A-D])\)?(?![A-Za-z])",
    r"answer is[:\s]*\(?([A-D])\)?(?![A-Za-z])",
    r"\banswer\b[:\s]*\(?([A-D])\)?(?![A-Za-z])",
    r"\boption\b[:\s]*\(?([A-D])\)?(?![A-Za-z])",
)

# Bare letter answer at the very start, e.g. "B." / "(C)" / "D".
_MMLU_LEADING_LETTER = r"^\(?([A-D])\)?(?:[\.\):,\-]|$)"
# "D A charismatic national leader": letter, whitespace, then free text. Only
# trusted when the text is that option's text (see _option_prefix_letter).
_MMLU_LEADING_LETTER_SPACED = r"^([A-D])[ \t]+(.+)"

# Option texts shorter than this are too easy to match by accident (e.g. "1").
_MIN_OPTION_TEXT_LEN = 4


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _option_texts(mmlu_example: dict[str, Any]) -> list[str]:
    options = mmlu_example.get("options") or []
    return [_normalize(str(option)) for option in options]


def _option_prefix_letter(tail: str, options: list[str]) -> str | None:
    """Return the letter whose option text starts `tail`, preferring the longest.

    This disambiguates cases where the model answered with the option *text*
    that happens to begin with a letter-like token, e.g. the generation
    "The answer is A national holiday." for option B = "A national holiday".
    """
    tail = _normalize(tail)
    best_letter, best_len = None, 0
    for letter, option in zip(MMLU_LETTERS, options):
        if len(option) >= _MIN_OPTION_TEXT_LEN and tail.startswith(option) and len(option) > best_len:
            best_letter, best_len = letter, len(option)
    return best_letter


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse a model generation into one of "A"/"B"/"C"/"D", else None."""
    if not model_output:
        return None

    # The zero-shot system prompt wraps the answer in a markdown code block and
    # then starts a new "# Query:" turn, so cut both off.
    text = model_output.split("# Query:")[0].replace("```", " ").strip()
    if not text:
        return None
    options = _option_texts(mmlu_example)

    # 1) Explicit "the correct answer is X" style statements.
    for pattern in _MMLU_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            # Guard against the letter actually being the first word of an
            # option's text rather than an option label.
            by_text = _option_prefix_letter(text[match.start(1) :], options)
            return by_text or letter

    # 2) Bare leading letter, e.g. "C)" or "B. ...".
    match = re.match(_MMLU_LEADING_LETTER, text)
    if match:
        return match.group(1).upper()

    # 3) "D A charismatic national leader" -- trust it only if the trailing text
    #    is really that option's text.
    match = re.match(_MMLU_LEADING_LETTER_SPACED, text)
    if match:
        letter, rest = match.group(1).upper(), match.group(2)
        index = MMLU_LETTERS.index(letter)
        if index < len(options) and _normalize(rest).startswith(options[index]):
            return letter

    # 4) No letter anywhere: fall back to quoting a single option verbatim.
    normalized = _normalize(text)
    hits = {
        letter
        for letter, option in zip(MMLU_LETTERS, options)
        if len(option) >= _MIN_OPTION_TEXT_LEN and option in normalized
    }
    if len(hits) == 1:
        return hits.pop()

    return None


# A number, optionally signed, with thousands separators and/or a decimal part.
# The lookbehind rejects a "-" glued to a preceding digit or ".", so the minus in
# "48-24" reads as subtraction rather than as the sign of "-24".
_GSM8K_NUMBER = r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?"


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a GSM8K generation into the last number it contains, else None."""
    if not model_output:
        return None
    numbers = re.findall(_GSM8K_NUMBER, model_output)
    if not numbers:
        return None
    # Drop thousands separators, plus any comma the greedy [\d,]* swallowed from
    # the following prose (e.g. "she had 1,000, so ..." -> "1,000,").
    return numbers[-1].replace(",", "")


def gsm8k_is_correct(prediction: str | None, gold: str) -> bool:
    """Compare a parsed GSM8K prediction against the gold answer numerically.

    String equality is too strict: the gold answers use thousands separators
    (e.g. "2,125") and a model may write "18.0" where the gold says "18".
    """
    if prediction is None:
        return False
    try:
        return float(prediction) == float(gold.replace(",", ""))
    except ValueError:
        return False
