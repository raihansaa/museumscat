#Deterministic, metric-justified string cleanup applied to final predictions.



from __future__ import annotations

from museumscat.config import MISSING


DANISH_FOLD = {
    "ö": "ø", "Ö": "Ø",
    "ä": "æ", "Ä": "Æ",
    "ü": "y", "Ü": "Y",
    "ó": "o", "ò": "o", "ú": "u",
    "φ": "ø",  # observed once: the model rendering ø as Greek phi
}

# Macron vowels only.
MACRON_FOLD = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
}
_MACRON_TABLE = str.maketrans(MACRON_FOLD)


MAX_PIPE_TOKENS = 6


def strip_macrons(text: str) -> str:
    """Replace macron vowels with their bare form, leaving Danish æ/ø/å untouched."""
    if not text:
        return text
    return text.translate(_MACRON_TABLE)


def danish_fold(text: str) -> str:
    """Map letters outside the Danish alphabet onto their Danish equivalents."""
    if not text or text == MISSING:
        return text
    return "".join(DANISH_FOLD.get(ch, ch) for ch in text)


def pipe_generously(text: str) -> str:
    """Split into pipe-separated tokens so the metric may pick the best ordering."""
    if not text or text == MISSING:
        return text
    tokens = [t for t in text.replace("|", " ").split() if t]
    if len(tokens) < 2 or len(tokens) > MAX_PIPE_TOKENS:
        return text
    return " | ".join(tokens)


def polish(text: str, use_pipes: bool = True) -> str:
    """All transforms, in the order they should be applied."""
    out = danish_fold(strip_macrons(text))
    return pipe_generously(out) if use_pipes else out
