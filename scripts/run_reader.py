#Run one VLM reader over the crops and cache every raw response.


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat.config import (  # noqa: E402
    DATE_COL, ID_COL, LOCALITY_COL, MISSING,
)
from museumscat.parsing import extract, parse_json  
from museumscat.readers import DEFAULT_MODEL, query

COLUMNS = (ID_COL, DATE_COL, LOCALITY_COL, "date_legibility", "locality_legibility", "raw")


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return set(df[ID_COL])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True, help="directory of cropped images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True, help="output CSV (resumable)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for reproducibility; >0 to sample a candidate pool")
    parser.add_argument("--limit", type=int, default=0, help="stop after N new reads")
    args = parser.parse_args(argv)

    crops = sorted(Path(args.crops).glob("*.jpeg"))
    if not crops:
        print(f"no crops found under {args.crops}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path)
    new_file = not out_path.exists()

    written = 0
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if new_file:
            writer.writerow(COLUMNS)
        for crop in crops:
            if crop.name in done:
                continue
            if args.limit and written >= args.limit:
                break
            try:
                raw = query(crop, args.model, temperature=args.temperature)
            except Exception as exc:  # keep a 3,300-image pass alive
                print(f"  {crop.name}: {exc}", file=sys.stderr)
                raw = ""
            date, locality, date_leg, loc_leg = extract(parse_json(raw))
            writer.writerow([crop.name, date or MISSING, locality or MISSING,
                             date_leg, loc_leg, raw])
            handle.flush()
            written += 1
            if written % 100 == 0:
                print(f"  {written} read", flush=True)

    print(f"done: {written} new reads, {len(done)} already cached -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
