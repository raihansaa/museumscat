#Faithful (UNOFFICIAL) implementation of the competition metric.

from __future__ import annotations

import itertools
import re
from typing import Sequence

_MISSING = "MISSING"

_DATE_SEP_RE = re.compile(r"[.,\-/·\s]+")
_WS_RE = re.compile(r"\s+")
_MAX_CARDS_PERMUTE = 6          # guard against factorial blow-up on pathological input


def levenshtein(a: str, b: str) -> int:
    """Edit distance with unit insert/delete/substitute costs (two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,         # deletion
                current[j - 1] + 1,      # insertion
                previous[j - 1] + cost,  # substitution
            ))
        previous = current
    return previous[-1]


def _canonicalize(text: str, is_date: bool) -> str:
    
    folded = text.replace("|", " ").casefold()
    if is_date:
        folded = _DATE_SEP_RE.sub(" ", folded)
    return _WS_RE.sub(" ", folded).strip()


def _ned(a: str, b: str) -> float:
    denom = max(len(a), len(b))
    return 0.0 if denom == 0 else levenshtein(a, b) / denom


def _is_missing(text: str) -> bool:
    return text.strip().casefold() == _MISSING.casefold()


def normalized_edit_distance(prediction: str, target: str, is_date: bool = False) -> float:
    
    pred = "" if prediction is None else str(prediction)
    tgt = "" if target is None else str(target)
    gt = "" if _is_missing(tgt) else _canonicalize(tgt, is_date)
    if _is_missing(pred):
        return _ned("", gt)
    cards = [c for c in (part.strip() for part in pred.split("|")) if c]
    if len(cards) <= 1 or len(cards) > _MAX_CARDS_PERMUTE:
        return _ned(_canonicalize(pred, is_date), gt)
    return min(_ned(_canonicalize(" ".join(order), is_date), gt)
               for order in itertools.permutations(cards))


def legacy_normalized_edit_distance(prediction: str, target: str,
                                    is_date: bool = False) -> float:
    
    pred = "" if prediction is None else str(prediction)
    tgt = "" if target is None else str(target)
    return _ned(pred, tgt)


def field_errors(predictions: Sequence[str], targets: Sequence[str],
                 is_date: bool = False) -> list[float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    return [normalized_edit_distance(p, t, is_date) for p, t in zip(predictions, targets)]


def aurc(errors: Sequence[float], confidences: Sequence[float]) -> float:
    
    if len(errors) != len(confidences):
        raise ValueError("errors and confidences must have equal length")
    n = len(errors)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    running = 0.0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        running += errors[idx]
        total += running / rank
    return total / n


def risk_at_coverage(errors: Sequence[float], confidences: Sequence[float],
                     coverage: float) -> float:
    """Mean error over the most-confident `coverage` fraction of examples."""
    n = len(errors)
    if n == 0:
        return 0.0
    k = max(1, round(coverage * n))
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)[:k]
    return sum(errors[i] for i in order) / k


def mean_aurc(errors_by_field: dict, confidences_by_field: dict) -> float:
    """Mean of per-field AURC across the scored fields."""
    parts = [aurc(errors_by_field[f], confidences_by_field[f]) for f in errors_by_field]
    return sum(parts) / len(parts)
