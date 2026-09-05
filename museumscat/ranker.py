"""The learned confidence ranker: predict a row's NED, use 1 - prediction as its confidence.

Mean AURC depends ONLY on the confidence *ranking*, so a per-field regressor that predicts
NED from features available at test time can reorder the predictions closer to the oracle
without changing a single transcription.

Design notes, all of which were decided by measurement at n=200:

* One model per field. Target = NED; confidence = ``1 - clip(pred, 0, 1)``.
* Features are computable from the reader's raw JSON alone, so re-assembling a submission
  is free -- no re-inference.
* Model = a seed-ensemble of shallow gradient-boosted trees. Averaging damps single-seed
  variance, which matters more than raw fit at this sample size. A linear model was worse.
* ``base_conf`` (the legibility-derived confidence) is a feature, so the ranker can only
  REFINE the working signal, never discard it.

**Validation is grouped by gold locality cluster.** Ungrouped folds leak badly: the same
place name recurs across many trays, so a row's near-duplicates land in the training half.

**And a warning this repository should carry loudly.** This ranker is a *challenger*. On
the hidden test set, wholesale confidence re-ranking went 0 for 8 -- including variants
that looked perfect locally. Fit it, look at it, learn from the features it selects, but
prefer the order-preserving demotions in ``cohorts.py`` for anything you actually ship.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from museumscat.config import MISSING, N_FOLDS, RANDOM_SEED

# Seeds are FIXED, not searched.
ENSEMBLE_SEEDS = tuple(range(12))
GBR_PARAMS = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8}

# Fixed feature order -- train/test parity depends on it, and base_conf stays at index 0.
DATE_FEATURES = ("base_conf", "is_missing", "length", "n_digits", "n_seg",
                 "has_year4", "has_month", "has_roman", "n_alpha_tok")
LOCALITY_FEATURES = ("base_conf", "is_missing", "length", "n_seg", "n_tok",
                     "avg_tok_len", "short", "has_digit", "has_colon",
                     "upper_ratio", "n_alpha_tok")

_YEAR_RE = re.compile(r"\b(1[789]\d\d|20[012]\d)\b")
_ALPHA_RE = re.compile(r"[a-zæøåäöü]+")
_DIGIT_RE = re.compile(r"\d")
SHORT_LOCALITY_LEN = 3

MONTHS = frozenset({
    "januar", "februar", "marts", "april", "maj", "juni", "juli", "august",
    "september", "oktober", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "okt", "nov", "dec",
})
ROMAN = frozenset({"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"})


def date_features(date: str, base_conf: float) -> dict[str, float]:
    """Shape features of a date string. Nothing here looks at the image."""
    missing = date == MISSING
    tokens = _ALPHA_RE.findall(date.lower())
    return {
        "base_conf": float(base_conf),
        "is_missing": float(missing),
        "length": 0.0 if missing else float(len(date)),
        "n_digits": 0.0 if missing else float(len(_DIGIT_RE.findall(date))),
        "n_seg": 0.0 if missing else float(date.count("|")),
        "has_year4": 0.0 if missing else float(bool(_YEAR_RE.search(date))),
        "has_month": float(any(t in MONTHS for t in tokens)),
        "has_roman": float(any(t in ROMAN for t in tokens)),
        "n_alpha_tok": float(len(tokens)),
    }


def locality_features(locality: str, base_conf: float) -> dict[str, float]:
    """Shape features of a locality string."""
    missing = locality == MISSING
    tokens = [] if missing else locality.split()
    alpha = _ALPHA_RE.findall(locality.lower())
    letters = sum(c.isalpha() for c in locality)
    return {
        "base_conf": float(base_conf),
        "is_missing": float(missing),
        "length": 0.0 if missing else float(len(locality)),
        "n_seg": 0.0 if missing else float(locality.count("|")),
        "n_tok": float(len(tokens)),
        "avg_tok_len": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "short": float((not missing) and len(locality.replace(" ", "")) <= SHORT_LOCALITY_LEN),
        "has_digit": 0.0 if missing else float(bool(_DIGIT_RE.search(locality))),
        "has_colon": 0.0 if missing else float(":" in locality),
        "upper_ratio": (sum(c.isupper() for c in locality) / letters) if letters else 0.0,
        "n_alpha_tok": float(len(alpha)),
    }


def feature_matrix(values: Sequence[str], base_confs: Sequence[float], is_date: bool):
    """Stack per-row feature dicts into an array in the fixed feature order."""
    import numpy as np

    builder = date_features if is_date else locality_features
    names = DATE_FEATURES if is_date else LOCALITY_FEATURES
    rows = [builder(str(v), c) for v, c in zip(values, base_confs)]
    return np.array([[row[name] for name in names] for row in rows], dtype=float)


def group_folds(groups: Sequence[str], n_folds: int = N_FOLDS, seed: int = RANDOM_SEED):
    """Grouped folds over gold locality clusters -- never split a cluster across folds."""
    import numpy as np

    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    rng.shuffle(unique)
    assignment = {g: i % n_folds for i, g in enumerate(unique)}
    fold_of = np.array([assignment[g] for g in groups])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_folds)]


def fit(features, targets):
    """Seed-ensemble of shallow GBRs. Returns a list of fitted models."""
    from sklearn.ensemble import GradientBoostingRegressor

    models = []
    for seed in ENSEMBLE_SEEDS:
        model = GradientBoostingRegressor(random_state=seed, **GBR_PARAMS)
        model.fit(features, targets)
        models.append(model)
    return models


def predict_confidence(models, features) -> list[float]:
    """Averaged predicted NED, turned into a confidence in [0, 1]."""
    import numpy as np

    predicted = np.mean([m.predict(features) for m in models], axis=0)
    return (1.0 - np.clip(predicted, 0.0, 1.0)).tolist()
