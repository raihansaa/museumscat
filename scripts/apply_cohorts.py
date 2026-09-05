#Demote rule-defined cohorts to the back of the ranking. Text is never touched.



from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat import metric  
from museumscat.cohorts import (  
    demote, error_rate, self_contradiction_mask, self_disagreement_mask, weight_still_held,
)
from museumscat.config import CONF_COLS, DATE_COL, ID_COL, LOCALITY_COL  


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="base submission CSV")
    parser.add_argument("--reads", required=True, help="reader CSV with a `raw` column")
    parser.add_argument("--reads-b", help="a SECOND draw of the same reader")
    parser.add_argument("--field", choices=("locality", "date"), default="locality")
    parser.add_argument("--disagreement", action="store_true", help="tier 1: unstable rows")
    parser.add_argument("--contradiction", action="store_true",
                        help="tier 2: MISSING while graded visible")
    parser.add_argument("--train", help="labelled CSV; enables the error-rate diagnostic")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import pandas as pd

    is_date = args.field == "date"
    col = DATE_COL if is_date else LOCALITY_COL
    conf_col = CONF_COLS[col]

    sub = pd.read_csv(args.submission, dtype=str, keep_default_na=False, encoding="utf-8")
    reads = pd.read_csv(args.reads, dtype=str, keep_default_na=False,
                        encoding="utf-8").set_index(ID_COL)
    ids = sub[ID_COL].tolist()
    raws = [reads.loc[i, "raw"] if i in reads.index else "" for i in ids]

    tiers, names = [], []
    if args.disagreement:
        if not args.reads_b:
            print("--disagreement needs --reads-b (a second draw)", file=sys.stderr)
            return 1
        reads_b = pd.read_csv(args.reads_b, dtype=str, keep_default_na=False,
                              encoding="utf-8").set_index(ID_COL)
        draw_a = [reads.loc[i, col] if i in reads.index else "" for i in ids]
        draw_b = [reads_b.loc[i, col] if i in reads_b.index else "" for i in ids]
        tiers.append(self_disagreement_mask(draw_a, draw_b, is_date))
        names.append("self-disagreement")
    if args.contradiction:
        tiers.append(self_contradiction_mask(sub[col].tolist(), raws, args.field))
        names.append("self-contradiction")
    if not tiers:
        print("nothing to do: pass --disagreement and/or --contradiction", file=sys.stderr)
        return 1

    conf = sub[conf_col].astype(float).tolist()

    errors = None
    if args.train:
        gold = pd.read_csv(args.train, dtype=str, keep_default_na=False,
                           encoding="utf-8").set_index(ID_COL)
        known = [i in gold.index for i in ids]
        if any(known):
            errors = metric.field_errors(
                sub[col].tolist(),
                [gold.loc[i, col] if i in gold.index else sub[col].iloc[k]
                 for k, i in enumerate(ids)],
                is_date,
            )

    for name, mask in zip(names, tiers):
        line = (f"{name:<20} {sum(mask):>5} rows  "
                f"weight held {weight_still_held(mask, conf):.2%}")
        if errors is not None:
            line += f"  error rate {error_rate(mask, errors):.0%}"
        print(line)

    sub[conf_col] = ["%.10g" % v for v in demote(conf, tiers)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"\ntext unchanged; {conf_col} permuted -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
