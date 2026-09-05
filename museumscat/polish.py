"""Deterministic, metric-justified string cleanup applied to final predictions.

Every transform here is safe *by construction* rather than by validation: it leaves the
confidence column untouched, so the AURC ranking is fixed, and it can only lower a row's
edit distance. AURC is monotone in per-row error under a fixed ranking, which makes these
the rare changes that cannot score worse -- unlike a reader swap, which rewrites the
predictions and moves the ranking underneath itself.

Three transforms, each with its measured justification:

1. DANISH ALPHABET FOLD. The data description states solutions are transcribed to
   "abcdefghijklmnopqrstuvwxyzæøå" and that ö -> ø. Checked against the labels: gold
   contains ZERO instances of ö/ä/ü, while our predictions carried 37 ö, 16 ü and 8 ä
   over 65 rows -- real Danish places the model spelled with German umlauts
   (Mön/Møn, Rönne/Rønne, Hövelte/Høvelte). Note the description ALSO claims "ÿ -> y",
   which is FALSE: gold contains ÿ three times ('Dÿrehaven', 'Jÿdske Aas Vendsÿssel'),
   so ÿ is deliberately preserved.

2. MACRON STRIP. Gemini's one systematic defect is inventing macrons (`Vamdrūp` for
   Vamdrup, `Fālster` for Falster) -- about 2% of strings, each turning an otherwise
   correct read wrong. The obvious fix is wrong: stripping every Unicode combining mark
   also destroys `Å` -> `A`, because Å decomposes to A + ring above, so `Århus` becomes
   `Arhus`. Danish æ/ø/å are LETTERS here, not accented variants, so the repair must
   NAME the characters it removes rather than removing a Unicode category.

3. PIPE GENEROSITY. From the data description: "The metric tries ALL possible orderings
   of the cards ... The pipes themselves are removed from the predictions, so you can use
   them generously when there is ambiguity." The metric therefore scores the BEST
   permutation of our pipe-separated tokens, and our existing word order is itself one of
   those permutations -- so splitting can only help or tie. Measured on the 200 labelled
   rows: 9 rows improved, 0 worsened, locality NED 0.3619 -> 0.3552.
"""

from __future__ import annotations

from museumscat.config import MISSING

# Non-Danish letters the readers emit, mapped to their Danish equivalents.
# y-with-diaeresis is deliberately ABSENT: it occurs in the ground truth and must survive.
DANISH_FOLD = {
    "ö": "ø", "Ö": "Ø",
    "ä": "æ", "Ä": "Æ",
    "ü": "y", "Ü": "Y",
    "ó": "o", "ò": "o", "ú": "u",
    "φ": "ø",  # observed once: the model rendering ø as Greek phi
}

# Macron vowels only. Deliberately not a category-based rule -- see the module docstring.
MACRON_FOLD = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
}
_MACRON_TABLE = str.maketrans(MACRON_FOLD)

# Permuting n cards costs n! comparisons; keep it bounded. Above this we leave the string
# alone, which is exactly what an unpiped prediction already does.
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
