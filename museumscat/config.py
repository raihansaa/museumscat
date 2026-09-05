"""Schema, conventions and paths for MuseumSCAT.

Every constant here comes from the competition files or from the verbatim conventions
observed in the 200 labelled rows of ``train.csv``. Paths are resolved from
``config.yaml`` at the repository root, or from the ``MUSEUMSCAT_DATA`` environment
variable, so the package runs against either the real competition data or the tiny
synthetic stand-ins in ``examples/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Schema ----------------------------------------------------------------------
ID_COL = "image_file"
DATE_COL = "verbatimDate"
DATE_CONF_COL = "verbatimDate_confidence"
LOCALITY_COL = "verbatimLocality"
LOCALITY_CONF_COL = "verbatimLocality_confidence"

FIELDS = (DATE_COL, LOCALITY_COL)
CONF_COLS = {DATE_COL: DATE_CONF_COL, LOCALITY_COL: LOCALITY_CONF_COL}
SUBMISSION_COLS = (ID_COL, DATE_COL, DATE_CONF_COL, LOCALITY_COL, LOCALITY_CONF_COL)

# --- Conventions (learned from the labels) -----------------------------------------
# The literal string the competition expects for an absent field. The metric compares
# it as the empty string, so emitting MISSING where gold has text costs NED 1.0 -- the
# maximum per-row loss, and the basis of the false-MISSING cohorts in cohorts.py.
MISSING = "MISSING"
MULTICARD_SEP = " | "
DANISH_ALPHABET = "abcdefghijklmnopqrstuvwxyzæøå"

# --- Cross-validation ---------------------------------------------------------------
# Folds are GROUPED by gold locality cluster. Ungrouped folds leak badly: the same
# place name recurs across many trays, so a row's near-duplicates land in the training
# half and every estimate comes out optimistic.
N_FOLDS = 5
RANDOM_SEED = 42


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is an optional convenience
        return {}
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def settings(config_path: Path | None = None) -> dict[str, Any]:
    """Merged settings: config.yaml overridden by MUSEUMSCAT_* environment variables."""
    cfg = _load_yaml(config_path or REPO_ROOT / "config.yaml")
    data = dict(cfg.get("data", {}))
    if os.environ.get("MUSEUMSCAT_DATA"):
        data["root"] = os.environ["MUSEUMSCAT_DATA"]
    cfg["data"] = data
    return cfg


def data_paths(config_path: Path | None = None) -> dict[str, Path]:
    """Resolve the four input paths the pipeline needs."""
    data = settings(config_path).get("data", {})
    root = Path(data.get("root", REPO_ROOT / "examples"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return {
        "root": root,
        "images": root / data.get("images", "images"),
        "train_csv": root / data.get("train_csv", "train.csv"),
        "test_csv": root / data.get("test_csv", "test.csv"),
    }
