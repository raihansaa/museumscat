"""Faithful (UNOFFICIAL) implementation of the competition metric.

The competition scores mean AURC over verbatimDate and verbatimLocality:
per-field error is the normalized edit distance between prediction and target,
and AURC is the area under the risk-coverage curve (examples ranked by the
predicted confidence, most-confident first). Lower is better.

Aligned to the official spec (Kaggle data description, 2026-07-16): NED is
case-INSENSITIVE, folds equivalent date separators, treats MISSING as an empty
string, and scores multi-card (pipe) predictions under their best ordering.

IMPORTANT: the organisers' metric_code/ is still the only source of truth.
Re-validate the moment it is available. Points most likely to differ, isolated
for easy alignment:
  * date separator set     -> _DATE_SEP_RE (we infer "/" beyond the documented set)
  * multi-card ordering     -> normalized_edit_distance() (we permute the prediction)
  * confidence-tie handling -> aurc()
"""
from __future__ import annotations

import itertools
import re
from typing import Sequence

# Literal sentinel for an absent field (mirrors config.MISSING). The official
# metric matches it as an empty string, so a missed present field scores NED 1.0
# and MISSING-vs-MISSING scores 0.0.
_MISSING = "MISSING"
# Date-only separator folding: the official metric treats these as equivalent
# ("27.5.2022" == "27,5,2022" == "27-5-2022" == "27·5·2022" == "27 5 2022").
# "/" (the common date slash) is added by inference. Isolated for easy correction
# once the organisers' metric_code/ is available.
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
    """Case-fold, drop card pipes, collapse whitespace; for dates, fold separators.

    Character conventions are preserved (e.g. 'aa' and 'ÿ' are NOT unicode-folded);
    only case and -- for dates -- separator punctuation are normalized, per the
    official metric.
    """
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
    """Official-aligned NED in [0, 1] (lower is better).

    Mirrors the competition metric (see reports/data_audit.md and the Kaggle data
    description): case-INSENSITIVE; date-separator folding when is_date; MISSING
    compared as empty; and multi-card scoring where the pipe-separated cards of the
    PREDICTION are tried in every order against the pipe-stripped target, best wins.
    """
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
    """Pre-alignment NED: plain levenshtein, case-EXACT, MISSING literal, no folding.

    This is NOT the competition metric -- use normalized_edit_distance for scoring.
    It exists only as a CONFIDENCE-RANKER TRAINING TARGET, because the ranker behind
    our best submission (LB 0.10743) was trained on these targets, and retraining the
    same ranker on the aligned targets REGRESSED it to 0.11924 on the leaderboard.
    The suspected cause is MISSING-as-empty: aligned targets score a missed present
    field at the full 1.0, so the ranker learns to distrust every MISSING prediction,
    but on test many MISSING predictions are correct. Keeping this available lets a
    challenger change ONE variable at a time against the 0.107 recipe.

    `is_date` is accepted and ignored (the legacy metric did no date folding), so the
    two functions are drop-in interchangeable at call sites.
    """
    pred = "" if prediction is None else str(prediction)
    tgt = "" if target is None else str(target)
    return _ned(pred, tgt)


def field_errors(predictions: Sequence[str], targets: Sequence[str],
                 is_date: bool = False) -> list[float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    return [normalized_edit_distance(p, t, is_date) for p, t in zip(predictions, targets)]


def aurc(errors: Sequence[float], confidences: Sequence[float]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Examples are ranked by confidence descending; risk(k) is the mean error of
    the k most-confident; AURC is the mean of risk(k) over all k. Ties keep
    input order (Python's sort is stable, including with reverse=True), giving
    a deterministic ranking.
    """
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
