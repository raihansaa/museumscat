#Crop every specimen photograph to its label band.


from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat.config import data_paths 
from museumscat.crop import CROP_MAX_DIM, crop_tag_region  

JPEG_QUALITY = 95


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="image file names (default: --all)")
    parser.add_argument("--all", action="store_true", help="crop every image in the data dir")
    parser.add_argument("--out", default="outputs/crops", help="output directory")
    parser.add_argument("--max-dim", type=int, default=CROP_MAX_DIM)
    parser.add_argument("--force", action="store_true", help="re-crop existing files")
    args = parser.parse_args(argv)

    paths = data_paths()
    names = args.images
    if args.all or not names:
        names = sorted(p.name for p in paths["images"].glob("*.jpeg"))
    if not names:
        print(f"no images found under {paths['images']}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods: Counter = Counter()
    done = skipped = 0
    start = time.time()

    for name in names:
        target = out_dir / name
        if target.exists() and not args.force:
            skipped += 1
            continue
        crop, method = crop_tag_region(paths["images"] / name, args.max_dim)
        methods[method] += 1
        crop.save(target, quality=JPEG_QUALITY)
        done += 1
        if done % 250 == 0:
            print(f"  {done} cropped ({done / (time.time() - start):.1f}/s)", flush=True)

    print(f"done: {done} cropped, {skipped} already present, {time.time() - start:.0f}s")
    if methods:
        print("crop methods:", dict(methods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
