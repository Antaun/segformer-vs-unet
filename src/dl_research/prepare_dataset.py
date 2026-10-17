"""Check (and optionally unpack) the LoveDA dataset, and write <out>/manifest.json.

    python -m dl_research.prepare_dataset --raw ~/Downloads/LoveDA --out data/LoveDA   # unzip Train/Val/Test.zip
    python -m dl_research.prepare_dataset --out data/LoveDA                            # verify an existing layout
    python -m dl_research.prepare_dataset --out data/synthetic --synthetic             # tiny fake dataset

This script never downloads anything; get the zips from the official LoveDA release (see README).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from dl_research.data import get_spec
from dl_research.utils import write_json

# Image counts as published for LoveDA; a mismatch only warns (re-verify against your copy of the release).
EXPECTED = {"Train": {"urban": 1156, "rural": 1366}, "Val": {"urban": 677, "rural": 992},
            "Test": {"urban": 883, "rural": 913}}


def prepare(out="data/LoveDA", raw=None, synthetic=False) -> dict:
    spec, out = get_spec("loveda"), Path(out)

    if synthetic:
        spec.make_synthetic(out)
    elif raw:
        for split in ("Train", "Val", "Test"):
            z = Path(raw) / f"{split}.zip"
            if z.exists():
                print(f"extracting {z} -> {out}")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
            else:
                print(f"warning: {z} not found, skipping")

    manifest, problems = {}, []
    for split in ("Train", "Val", "Test"):
        try:
            samples = spec.scan(out, split, require_labels=split != "Test")
        except FileNotFoundError as e:
            problems.append(str(e))
            continue
        counts = {d: sum(s.domain == d for s in samples) for d in spec.domains}
        manifest[split] = {"counts": counts, "labeled": all(s.mask for s in samples)}
        if not synthetic and counts != EXPECTED[split]:
            print(f"warning: {split} counts {counts} differ from expected {EXPECTED[split]}")
        if split != "Test":  # sanity-check label values on a sample of masks
            vals = set()
            for s in samples[:: max(1, len(samples) // 50)]:
                vals |= set(np.unique(np.asarray(Image.open(s.mask))).tolist())
            manifest[split]["raw_mask_values_seen"] = sorted(vals)
            if not vals <= set(range(8)):
                print(f"warning: {split} masks contain values outside 0..7: {sorted(vals - set(range(8)))}")
        print(f"{split}: {counts}")
    if problems:
        raise SystemExit("\n".join(problems))
    write_json(out / "manifest.json", manifest)
    print(f"ok -> {out / 'manifest.json'}")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/LoveDA")
    ap.add_argument("--raw", help="folder holding Train.zip / Val.zip / Test.zip")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)
    prepare(args.out, args.raw, args.synthetic)


if __name__ == "__main__":
    main()
