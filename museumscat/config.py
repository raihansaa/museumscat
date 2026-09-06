#Schema, conventions and paths for MuseumSCAT.

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

MISSING = "MISSING"
MULTICARD_SEP = " | "
DANISH_ALPHABET = "abcdefghijklmnopqrstuvwxyzæøå"

# --- Cross-validation ---------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
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
