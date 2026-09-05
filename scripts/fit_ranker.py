#Fit the confidence ranker and report grouped out-of-fold AURC against the base column.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat import metric, ranker  # noqa: E402
from museumscat.config import DATE_COL, ID_COL, LOCALITY_COL  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--train", required=True, help="labelled CSV with gold columns")
    parser.add_argument("--field", choices=("locality", "date"), default="locality")
    parser.add_argument("--out", help="optional CSV of fitted confidences")
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd

    is_date = args.field == "date"
    col = DATE_COL if is_date else LOCALITY_COL
    base_conf_col = f"{args.field}_base_conf"

    ens = pd.read_csv(args.ensemble, dtype=str, keep_default_na=False, encoding="utf-8")
    gold = pd.read_csv(args.train, dtype=str, keep_default_na=False,
                       encoding="utf-8").set_index(ID_COL)
    ens = ens[ens[ID_COL].isin(gold.index)].reset_index(drop=True)
    if ens.empty:
        print("no labelled rows in the ensemble file", file=sys.stderr)
        return 1

    predictions = ens[col].tolist()
    targets = [gold.loc[i, col] for i in ens[ID_COL]]
    errors = metric.field_errors(predictions, targets, is_date)
    base_conf = ens[base_conf_col].astype(float).tolist()

    features = ranker.feature_matrix(predictions, base_conf, is_date)
    # Group by gold string: identical localities are near-duplicate rows.
    folds = ranker.group_folds([str(t).casefold() for t in targets])

    oof = np.zeros(len(ens))
    for train_idx, test_idx in folds:
        models = ranker.fit(features[train_idx], np.asarray(errors)[train_idx])
        oof[test_idx] = ranker.predict_confidence(models, features[test_idx])

    base_aurc = metric.aurc(errors, base_conf)
    oof_aurc = metric.aurc(errors, oof.tolist())
    oracle = metric.aurc(errors, [-e for e in errors])
    random_aurc = sum(errors) / len(errors)

    print(f"field: {args.field}   n = {len(ens)}")
    print(f"  random ordering (= mean NED)   {random_aurc:.5f}")
    print(f"  base (legibility) ordering     {base_aurc:.5f}")
    print(f"  ranker, grouped OOF            {oof_aurc:.5f}")
    print(f"  oracle ordering, same text     {oracle:.5f}")
    span = random_aurc - oracle
    if span > 0:
        print(f"  ranker captures {100 * (random_aurc - oof_aurc) / span:.0f}% "
              f"of the random -> oracle span")
    if oof_aurc > base_aurc:
        print("  NOTE: the ranker is WORSE than the base column here. "
              "This happened to us on the leaderboard 8 times out of 8.")

    if args.out:
        ens[f"{args.field}_ranker_conf"] = oof
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        ens.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
