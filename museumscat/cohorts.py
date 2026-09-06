#Cohort demotion: the only ordering operator that ever worked.


from __future__ import annotations

from collections.abc import Sequence

from museumscat.consensus import canonical
from museumscat.parsing import is_missing, legibility

# A grade meaning "the reader saw no label at all". Anything else asserts a label WAS
# visible -- so pairing it with a MISSING value is a self-contradiction.
ABSENT_GRADES = ("absent", "none", "")


def demote(confidences: Sequence[float], tiers: Sequence[Sequence[bool]]) -> list[float]:
   
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

    return [
        is_missing(value) and legibility(raw, field) not in ABSENT_GRADES
        for value, raw in zip(values, raws)
    ]


def self_disagreement_mask(
    draw_a: Sequence[str], draw_b: Sequence[str], is_date: bool = False
) -> list[bool]:
   
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
