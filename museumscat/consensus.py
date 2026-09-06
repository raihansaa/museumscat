"""Combining several readers: agreement strata, majority vote, and frozen-rank substitution.

This module holds the campaign's main finding, so it is worth stating plainly.

**A stronger reader should SUBSTITUTE, not VOTE.** Adding a fifth, better reader as an
extra vote cannot overturn a string that already holds a plurality. That is structural,
not a tuning problem, and it locks the newcomer out of the 2-1-1 stratum which
measurement says is the single largest prize:
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from museumscat import metric
from museumscat.config import MISSING
from museumscat.polish import polish
from museumscat.parsing import is_missing


def canonical(text: object, is_date: bool = False) -> str:
    #Metric-canonical form of a string, for comparing two readers' answers.


    return metric._canonicalize(polish(str(text)), is_date)


def stratum(answers: Sequence[str], is_date: bool = False) -> str:

    counts = sorted(Counter(canonical(a, is_date) for a in answers).values(), reverse=True)
    if len(counts) == 1:
        return f"{counts[0]}-0"
    return "-".join(str(c) for c in counts)


def majority_vote(answers: Sequence[str], is_date: bool = False) -> str:

    answers = [str(a) for a in answers if str(a).strip()]
    if not answers:
        return MISSING
    counts = Counter(canonical(a, is_date) for a in answers)
    best = max(counts.values())
    for answer in answers:  # stable: first reader with a top-count answer wins
        if counts[canonical(answer, is_date)] == best:
            return answer
    return answers[0]


def agreement_fraction(answer: str, pool: Iterable[str], is_date: bool = False) -> float:

    values = [str(v) for v in pool]
    if not values:
        return 0.0
    target = canonical(answer, is_date)
    return sum(canonical(v, is_date) == target for v in values) / len(values)


def substitute(
    base: Sequence[str],
    challenger: Sequence[str],
    strata: Sequence[str],
    allowed_strata: Iterable[str],
    *,
    keep_missing: bool = True,
) -> list[str]:

    allowed = set(allowed_strata)
    if not (len(base) == len(challenger) == len(strata)):
        raise ValueError("base, challenger and strata must have equal length")

    out = []
    for current, candidate, row_stratum in zip(base, challenger, strata):
        take = row_stratum in allowed and str(candidate).strip() != ""
        if take and keep_missing and is_missing(candidate) and not is_missing(current):
            take = False
        out.append(str(candidate) if take else str(current))
    return out


SHIPPED_SUBSTITUTION_STRATA = ("2-1-1", "2-2", "1-1-1-1")
