"""Cohort demotion: the only ordering operator that ever worked.

Across 85 submissions on the hidden test set:

    order-preserving cohort demotion    8 for 8
    text substitution                   4 for 4
    wholesale confidence re-ranking     0 for 8

Every attempt to replace the confidence column with a better-looking one regressed,
including two that reached P(better) = 1.000 on the labelled set. Both a trained
re-ranker and an untrained heuristic one cost about +0.011 on their field. Moving a
rule-defined cohort to the back while leaving every other row in its existing relative
position worked every single time.

Our reading of why: the deployed order already encodes a great deal of fitted signal, and
a replacement ordering has to beat all of it at once. A demotion only has to be right
about the rows it touches.

``demote`` is therefore deliberately narrow. It permutes the confidence column among its
own values -- the multiset of confidences is unchanged, only their assignment moves -- so
the result is guaranteed to be a valid confidence column and the operation is block
displacement rather than re-ranking.

TWO RULES FOR CHOOSING A COHORT
-------------------------------
**Measure its error rate first.** A 100%-error cohort pays. Cohorts measured at 25%, 15%
and 11% all COST score. This is the single cheapest check available and it is free.

**Rank by expected error MAGNITUDE, not probability.** AURC charges a wrong row by how
wrong it is. A false MISSING scores NED 1.0 -- the maximum -- and our legibility mapping
hands "absent" a confidence of 0.85, the second-highest band. A maximal error carried at
high rank is precisely the shape the metric punishes hardest, which is why the
self-contradiction cohort was worth more than every probability-based cohort combined.

BREAK-EVEN, which is what makes a small cohort shippable
--------------------------------------------------------
Demoting MISSING rows *blind* fails (+0.00167 locality, +0.00057 date) because correct
abstentions outnumber false ones 30:6 and 59:5. The arithmetic of those same counts gives
the precision a demoted row actually needs:

    locality   break-even 29%   base rate 17%
    date       break-even  9%   base rate 7.8%

Compare the ~98% a general-purpose verifier needs. Price a lever against its ACTUAL
break-even, not a guessed bar -- we abandoned a usable instrument for missing a
requirement that did not apply to it.
"""

from __future__ import annotations

from collections.abc import Sequence

from museumscat.consensus import canonical
from museumscat.parsing import is_missing, legibility

# A grade meaning "the reader saw no label at all". Anything else asserts a label WAS
# visible -- so pairing it with a MISSING value is a self-contradiction.
ABSENT_GRADES = ("absent", "none", "")


def demote(confidences: Sequence[float], tiers: Sequence[Sequence[bool]]) -> list[float]:
    """Move each tier to the bottom of the ranking, in order, preserving inner order.

    `tiers` are ordered by severity. A row matching several tiers falls to the DEEPEST one
    it matches, since a row that is both unstable AND self-contradictory is the more severe
    case -- so assignment runs from the last tier backwards.

    The returned column is a permutation of the input values: same multiset, new
    assignment. Nothing is invented, nothing is rescaled, and no text changes.
    """
    import numpy as np

    conf = np.asarray(confidences, dtype=float)
    n = len(conf)
    if any(len(t) != n for t in tiers):
        raise ValueError("every tier mask must match the confidence column length")

    # Current ranking: confidence descending, ties broken by original position.
    order = np.lexsort((np.arange(n), -conf))

    assigned = np.zeros(n, dtype=bool)
    groups: list = [None] * len(tiers)
    for k in range(len(tiers) - 1, -1, -1):
        group = np.asarray(tiers[k], dtype=bool) & ~assigned
        assigned |= group
        groups[k] = group

    new_order = np.concatenate(
        [order[~assigned[order]]] + [order[g[order]] for g in groups]
    )
    values = np.sort(conf)[::-1]
    out = np.empty(n, dtype=float)
    out[new_order] = values
    return out.tolist()


def self_contradiction_mask(values: Sequence[str], raws: Sequence[str], field: str) -> list[bool]:
    """Rows whose value is MISSING while the reader graded the field as visible.

    The prompt states: if a field is absent, output MISSING *and* grade it "absent". A row
    returning MISSING while grading the field clear/partial/unreadable has broken its own
    instruction -- it asserts a label is present AND that there is no value on it.

    This is a LOGICAL guarantee, not a statistical enrichment, which is why it survived
    n=8 when every previous small-n cohort did not. On the labelled set it fired 5/200
    (locality) and 3/200 (date): all 8 wrong, all NED 1.000, all with a real gold value.
    """
    return [
        is_missing(value) and legibility(raw, field) not in ABSENT_GRADES
        for value, raw in zip(values, raws)
    ]


def self_disagreement_mask(
    draw_a: Sequence[str], draw_b: Sequence[str], is_date: bool = False
) -> list[bool]:
    """Rows where the SAME reader gave different answers on two independent draws.

    gemini-3.7-flash is not deterministic at temperature 0 -- only 34 of 40 localities
    reproduced on a byte-identical re-read. Demoting the rows that did not reproduce paid
    -0.00132 on the leaderboard.

    Note the comparison is metric-canonical, so a purely cosmetic difference does not
    count as disagreement. That matters: demoting COSMETIC disagreements measurably COST
    score (+0.00081 on the date field).

    This is the K=2 form. Sampling K>=3 at non-zero temperature would let the pool
    *resolve* the disagreement rather than only detect it; we never bought it.
    """
    return [
        canonical(a, is_date) != canonical(b, is_date)
        for a, b in zip(draw_a, draw_b)
    ]


def error_rate(mask: Sequence[bool], errors: Sequence[float], threshold: float = 0.0) -> float:
    """Share of a cohort that is actually wrong. Measure this BEFORE shipping the cohort.

    A 100%-error cohort pays. Cohorts at 25%, 15% and 11% all cost score.
    """
    rows = [e for m, e in zip(mask, errors) if m]
    if not rows:
        return 0.0
    return sum(e > threshold for e in rows) / len(rows)


def weight_still_held(mask: Sequence[bool], confidences: Sequence[float]) -> float:
    """Share of total AURC weight the cohort still carries at its current ranking.

    This is the size of the prize, and it is also a kill criterion: a cohort already
    sitting at the bottom of the order holds no weight, so demoting it further can win
    nothing no matter how precise the rule is. We killed the date arm of one cohort on
    exactly this -- its rows already sat at the 97th percentile holding 0.16% of weight.
    """
    import numpy as np

    conf = np.asarray(confidences, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0
    # AURC weight of rank k is proportional to the tail sum of 1/j, j >= k.
    weights = np.cumsum(1.0 / np.arange(n, 0, -1))[::-1]
    ranks = np.empty(n, dtype=int)
    ranks[np.lexsort((np.arange(n), -conf))] = np.arange(n)
    mask_arr = np.asarray(mask, dtype=bool)
    return float(weights[ranks[mask_arr]].sum() / weights.sum())
