#Assemble the final submission and validate it before upload.



from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat.config import (
    DATE_COL, ID_COL, LOCALITY_COL, data_paths,
)
from museumscat.submission import assemble, compare, validate, write


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", help="ensemble/substituted CSV to assemble from")
    parser.add_argument("--test", help="test.csv (defaults to the configured data dir)")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="diff two submissions instead of building one")
    args = parser.parse_args(argv)

    if args.compare:
        for key, value in compare(*args.compare).items():
            print(f"  {key:<32} {value}")
        return 0

    if not (args.ensemble and args.out):
        parser.error("--ensemble and --out are required unless --compare is used")

    import pandas as pd

    ens = pd.read_csv(args.ensemble, dtype=str, keep_default_na=False, encoding="utf-8")
    df = assemble(
        ens[ID_COL],
        ens[DATE_COL],
        ens["date_base_conf"],
        ens[LOCALITY_COL],
        ens["locality_base_conf"],
    )
    path = write(df, args.out)

    test_path = Path(args.test) if args.test else data_paths()["test_csv"]
    if test_path.exists():
        errors = validate(path, test_path)
        if errors:
            print("FAIL")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("PASS")
    else:
        print(f"(skipped validation: no test file at {test_path})")

    print(f"{len(df)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
