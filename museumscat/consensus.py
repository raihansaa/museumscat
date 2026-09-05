"""Combining several readers: agreement strata, majority vote, and frozen-rank substitution.

This module holds the campaign's main finding, so it is worth stating plainly.

**A stronger reader should SUBSTITUTE, not VOTE.** Adding a fifth, better reader as an
extra vote cannot overturn a string that already holds a plurality. That is structural,
not a tuning problem, and it locks the newcomer out of the 2-1-1 stratum -- which
measurement says is the single largest prize:

    stratum      dAURC from substitution   P(better)   reachable by a 5th vote?
    4-0                  -0.0015             0.534     no
    3-1                  -0.0027             0.725     no
    2-2                  -0.0027             0.855     yes
    1-1-1-1              -0.0156             0.997     yes
    2-1-1                -0.0185             0.999     NO      <- the largest, and unreachable

**The value of a stronger reader falls monotonically with agreement, and reverses.** On
held-out data: complete disagreement -0.00309, 2-1-1/2-2 -0.00030, and 3-1 **+0.00164 --
actively harmful**. Substituting into rows where three of four readers already agree makes
the submission worse, so substitution must be restricted to strata that clear a bar.

**Substitution is frozen-rank.** It replaces text and leaves the confidence column exactly
as it was. Better text you cannot rank is worth nothing -- we measured that directly: a
reader that transcribed dates *better* still lost score when swapped in, because the date
ranker's ordering was calibrated against the previous reader's error pattern.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from museumscat import metric
from museumscat.config import MISSING
from museumscat.polish import polish
from museumscat.parsing import is_missing


def canonical(text: object, is_date: bool = False) -> str:
    """Metric-canonical form of a string, for comparing two readers' answers.

    Uses the same canonicalisation the metric applies, so two answers count as
    "agreeing" exactly when the metric would score them identically.
    """
    return metric._canonicalize(polish(str(text)), is_date)


def stratum(answers: Sequence[str], is_date: bool = False) -> str:
    """Agreement pattern of N readers, as a dash-joined descending count.

    Four readers give "4-0" (unanimous), "3-1", "2-2", "2-1-1" or "1-1-1-1". The pattern,
    not the answers, is what predicts risk: two disagreement strata held 106 of 200
    labelled rows and 83.9% of the locality AURC loss, while unanimous reads took 27.5%
    of the metric's weight and produced 5.9% of its loss.
    """
    counts = sorted(Counter(canonical(a, is_date) for a in answers).values(), reverse=True)
    if len(counts) == 1:
        return f"{counts[0]}-0"
    return "-".join(str(c) for c in counts)


def majority_vote(answers: Sequence[str], is_date: bool = False) -> str:
    """Plurality answer, ties broken by reader order (earlier readers are stronger).

    Returns the ORIGINAL string of the winning group, not its canonical form -- the task
    is verbatim, so the submitted text must keep its own spelling and punctuation.
    """
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
    """Share of the pool that agrees with `answer` under metric canonicalisation."""
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
    """Frozen-rank text substitution, restricted to the strata that clear the bar.

    Replaces base text with the challenger's read wherever the row's agreement stratum is
    in `allowed_strata`. The confidence column is NOT passed in and NOT touched: this is a
    text-only operator, which is what makes it composable with the ordering operators in
    cohorts.py.

    `keep_missing` refuses a substitution that would turn a real answer into MISSING. A
    false MISSING is NED 1.0 -- the maximum per-row loss -- so a challenger abstaining is
    never worth taking over an existing answer.
    """
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


# The strata that cleared the bootstrap bar and shipped. 3-1 is deliberately excluded:
# it measured +0.00164 on held-out data, i.e. substituting there makes things worse.
SHIPPED_SUBSTITUTION_STRATA = ("2-1-1", "2-2", "1-1-1-1")
