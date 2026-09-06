#Assemble, write and validate a submission file.


from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from museumscat.config import (
    CONF_COLS, DATE_COL, DATE_CONF_COL, ID_COL, LOCALITY_COL,
    LOCALITY_CONF_COL, MISSING, SUBMISSION_COLS,
)


def assemble(ids, date_pred, date_conf, loc_pred, loc_conf):
    """Build the submission frame in the exact competition column order."""
    import pandas as pd

    return pd.DataFrame({
        ID_COL: list(ids),
        DATE_COL: list(date_pred),
        DATE_CONF_COL: list(date_conf),
        LOCALITY_COL: list(loc_pred),
        LOCALITY_CONF_COL: list(loc_conf),
    })


def write(df, path: str | Path) -> Path:
    """Write a submission, UTF-8, minimal quoting, LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(SUBMISSION_COLS)].to_csv(
        path, index=False, encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL, lineterminator="\n",
    )
    return path


def validate(submission_path: str | Path, test_path: str | Path) -> list[str]:

    import pandas as pd

    errors: list[str] = []
    sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False, encoding="utf-8")
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8")

    for col in SUBMISSION_COLS:
        if col not in sub.columns:
            errors.append(f"missing column: {col}")
    if errors:
        return errors

    sub_ids = sub[ID_COL].tolist()
    test_ids = set(test[ID_COL])
    if len(sub_ids) != len(set(sub_ids)):
        errors.append("duplicate image_file rows")
    missing_ids = test_ids - set(sub_ids)
    extra_ids = set(sub_ids) - test_ids
    if missing_ids:
        errors.append(f"{len(missing_ids)} test rows missing (e.g. {sorted(missing_ids)[:3]})")
    if extra_ids:
        errors.append(f"{len(extra_ids)} unknown image_file rows (e.g. {sorted(extra_ids)[:3]})")

    for conf_col in (DATE_CONF_COL, LOCALITY_CONF_COL):
        for value in sub[conf_col]:
            try:
                number = float(value)
            except ValueError:
                errors.append(f"non-numeric confidence in {conf_col}: {value!r}")
                break
            if math.isnan(number) or math.isinf(number) or not 0.0 <= number <= 1.0:
                errors.append(f"confidence out of [0,1] in {conf_col}: {value!r}")
                break

    for col in (DATE_COL, LOCALITY_COL):
        if (sub[col].str.len() == 0).any():
            errors.append(f"empty prediction in {col} (use '{MISSING}')")

    return errors


def compare(path_a: str | Path, path_b: str | Path) -> dict:

    import pandas as pd

    a = pd.read_csv(path_a, dtype=str, keep_default_na=False, encoding="utf-8")
    b = pd.read_csv(path_b, dtype=str, keep_default_na=False, encoding="utf-8")
    report = {
        "rows": (len(a), len(b)),
        "ids_aligned": a[ID_COL].tolist() == b[ID_COL].tolist(),
    }
    for col in (DATE_COL, LOCALITY_COL):
        report[col] = "identical" if a[col].tolist() == b[col].tolist() else "DIFFERS"
    for field, conf_col in CONF_COLS.items():
        differing = int((a[conf_col] != b[conf_col]).sum())
        report[conf_col] = f"differs on {differing} rows ({differing / max(len(a), 1):.1%})"
    return report
