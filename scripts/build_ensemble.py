#Combine several readers into base text, an agreement stratum, and a base confidence.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat.config import DATE_COL, ID_COL, LOCALITY_COL  
from museumscat.consensus import majority_vote, stratum  
from museumscat.polish import polish  
from museumscat.prompts import confidence_of


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read", action="append", required=True,
                        help="reader CSV from run_reader.py (repeat, strongest first)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-pipes", action="store_true",
                        help="skip pipe-generous splitting in the polish step")
    args = parser.parse_args(argv)

    import pandas as pd

    frames = [
        pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8").set_index(ID_COL)
        for p in args.read
    ]
    ids = list(frames[0].index)

    rows = []
    for image_id in ids:
        dates = [f.loc[image_id, DATE_COL] for f in frames if image_id in f.index]
        locs = [f.loc[image_id, LOCALITY_COL] for f in frames if image_id in f.index]
        date_legs = [f.loc[image_id, "date_legibility"] for f in frames if image_id in f.index]
        loc_legs = [f.loc[image_id, "locality_legibility"] for f in frames if image_id in f.index]

        rows.append({
            ID_COL: image_id,
            DATE_COL: polish(majority_vote(dates, is_date=True), use_pipes=False),
            LOCALITY_COL: polish(majority_vote(locs), use_pipes=not args.no_pipes),
            "date_stratum": stratum(dates, is_date=True),
            "locality_stratum": stratum(locs),
            "date_base_conf": confidence_of(date_legs[0] if date_legs else ""),
            "locality_base_conf": confidence_of(loc_legs[0] if loc_legs else ""),
        })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")

    print(f"{len(out)} rows -> {args.out}")
    print("\nlocality strata:")
    for name, count in out["locality_stratum"].value_counts().items():
        print(f"  {name:<10} {count:>5}  ({count / len(out):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
