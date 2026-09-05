#Frozen-rank text substitution: let a stronger reader REPLACE text in chosen strata.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat.config import DATE_COL, ID_COL, LOCALITY_COL  
from museumscat.consensus import SHIPPED_SUBSTITUTION_STRATA, substitute 
from museumscat.polish import polish  


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True, help="output of build_ensemble.py")
    parser.add_argument("--challenger", required=True, help="reader CSV to substitute from")
    parser.add_argument("--field", choices=("locality", "date"), default="locality")
    parser.add_argument("--strata", nargs="*", default=list(SHIPPED_SUBSTITUTION_STRATA))
    parser.add_argument("--allow-missing", action="store_true",
                        help="permit a substitution that turns an answer into MISSING")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import pandas as pd

    is_date = args.field == "date"
    col = DATE_COL if is_date else LOCALITY_COL
    stratum_col = f"{args.field}_stratum"

    base = pd.read_csv(args.ensemble, dtype=str, keep_default_na=False, encoding="utf-8")
    challenger = pd.read_csv(args.challenger, dtype=str, keep_default_na=False,
                             encoding="utf-8").set_index(ID_COL)

    candidate = [
        polish(challenger.loc[i, col], use_pipes=not is_date) if i in challenger.index else ""
        for i in base[ID_COL]
    ]
    updated = substitute(
        base[col].tolist(), candidate, base[stratum_col].tolist(), args.strata,
        keep_missing=not args.allow_missing,
    )

    changed = sum(a != b for a, b in zip(base[col].tolist(), updated))
    base[col] = updated
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")

    print(f"strata substituted: {', '.join(args.strata)}")
    print(f"{changed} of {len(base)} {args.field} values changed -> {args.out}")
    print("confidence column untouched (frozen rank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
