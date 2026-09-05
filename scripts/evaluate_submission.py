#Score a submission locally against labelled rows, and print the ordering decomposition.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat import metric 
from museumscat.config import CONF_COLS, DATE_COL, ID_COL, LOCALITY_COL  


def _decompose(errors, confidences, label: str) -> float:
    random_aurc = sum(errors) / len(errors)
    ours = metric.aurc(errors, confidences)
    oracle = metric.aurc(errors, [-e for e in errors])
    span = random_aurc - oracle
    print(f"\n{label}  (n = {len(errors)})")
    print(f"  random ordering (= mean NED)  {random_aurc:.5f}")
    print(f"  your ordering                 {ours:.5f}", end="")
    if span > 0:
        print(f"   ({100 * (random_aurc - ours) / span:.0f}% of the span)")
    else:
        print()
    print(f"  oracle ordering, same text    {oracle:.5f}")
    return ours


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--train", required=True)
    args = parser.parse_args(argv)

    import pandas as pd

    sub = pd.read_csv(args.submission, dtype=str, keep_default_na=False, encoding="utf-8")
    gold = pd.read_csv(args.train, dtype=str, keep_default_na=False,
                       encoding="utf-8").set_index(ID_COL)
    sub = sub[sub[ID_COL].isin(gold.index)].reset_index(drop=True)
    if sub.empty:
        print("no labelled rows in the submission", file=sys.stderr)
        return 1

    scores = {}
    all_errors = {}
    for col, label in ((LOCALITY_COL, "locality"), (DATE_COL, "date")):
        targets = [gold.loc[i, col] for i in sub[ID_COL]]
        errors = metric.field_errors(sub[col].tolist(), targets, col == DATE_COL)
        conf = sub[CONF_COLS[col]].astype(float).tolist()
        scores[label] = _decompose(errors, conf, label)
        all_errors[label] = (errors, conf)

    print(f"\nMEAN AURC  {sum(scores.values()) / len(scores):.5f}")

    print("\nreading counterfactual (ordering held fixed):")
    for fraction in (0.0, 0.25, 0.50, 0.75):
        total = 0.0
        for errors, conf in all_errors.values():
            order = sorted(range(len(errors)), key=lambda i: -errors[i])
            fixed = list(errors)
            for i in order[:int(round(fraction * sum(e > 0 for e in errors)))]:
                fixed[i] = 0.0
            total += metric.aurc(fixed, conf)
        print(f"  fix {fraction:>4.0%} of wrong rows  ->  {total / len(all_errors):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
