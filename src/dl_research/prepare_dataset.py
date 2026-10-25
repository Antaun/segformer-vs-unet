"""Check, unpack, and (opt-in) download the LoveDA dataset, and write <out>/manifest.json.

    python -m dl_research.prepare_dataset --raw ~/Downloads/LoveDA --out data/LoveDA   # unzip Train/Val/Test.zip
    python -m dl_research.prepare_dataset --out data/LoveDA                            # verify an existing layout
    python -m dl_research.prepare_dataset --out data/LoveDA --download --splits Train Val   # fetch from Zenodo
    python -m dl_research.prepare_dataset --out data/synthetic --synthetic             # tiny fake dataset

Nothing is downloaded unless you pass --download (or call `download_loveda`).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from dl_research.data import get_spec
from dl_research.utils import write_json

SPLITS = ("Train", "Val", "Test")
# Image counts of the official release as seen in a local copy; a mismatch only warns.
EXPECTED = {"Train": {"urban": 1156, "rural": 1366}, "Val": {"urban": 677, "rural": 992},
            "Test": {"urban": 820, "rural": 976}}
# Official files, Zenodo record 5706578 (open access). Sizes and MD5s are from the record's metadata.
# The record's metadata says CC-BY-4.0 but its description says CC BY-NC-SA 4.0 (non-commercial): check before use.
ZENODO = {
    "Train": ("https://zenodo.org/api/records/5706578/files/Train.zip/content", "de2b196043ed9b4af1690b3f9a7d558f", 4021669263),
    "Val": ("https://zenodo.org/api/records/5706578/files/Val.zip/content", "84cae2577468ff0b5386758bb386d31d", 2425958254),
    "Test": ("https://zenodo.org/api/records/5706578/files/Test.zip/content", "a489be0090465e01fb067795d24e6b47", 3126023212),
}


def md5sum(path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest, md5: str) -> None:
    """Resume and retry interrupted curl transfers, then verify the download with MD5."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5sum(dest) == md5:
        print(f"[ok] {dest.name} already downloaded and verified")
        return
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(dest), url],
        check=True,
    )
    if md5sum(dest) != md5:
        raise RuntimeError(f"MD5 mismatch for {dest}; delete it and retry")
    print(f"[ok] {dest.name} downloaded and verified")


def organize(src, out) -> dict:
    """Find every <Split>/ directory (containing Urban/ and Rural/) under `src`, wherever it is nested, and move
    it to <out>/<Split>. Handles `Train/Urban`, `Train/Train/Urban` and similar re-packagings. Existing
    destinations are left alone. `src` must differ from `out`."""
    src, out = Path(src), Path(out)
    found: dict[str, Path] = {}
    for d, dirs, _ in os.walk(src):
        if {"Urban", "Rural"} <= set(dirs) and Path(d).name in SPLITS:
            found.setdefault(Path(d).name, Path(d))
    for split, d in found.items():
        dest = out / split
        if dest.exists():
            print(f"[skip] {dest} already exists")
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
    return found


def download_loveda(out="data/LoveDA", splits=("Train", "Val"), work=None, keep_zips=False) -> None:
    """Download the official zips from Zenodo (MD5-verified), extract, and arrange as <out>/<Split>/{Urban,Rural}/...
    Test.zip (3.1 GB) has no labels and is not needed for the study, so it is not in the default `splits`."""
    out = Path(out)
    work = Path(work) if work else out.parent / f"{out.name}_download"
    for split in splits:
        if (out / split).exists():
            print(f"[skip] {out / split} already exists")
            continue
        url, md5, _ = ZENODO[split]
        zpath = work / f"{split}.zip"
        download_file(url, zpath, md5)
        ex = work / f"extracted_{split}"
        print(f"extracting {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ex)
        organize(ex, out)
        shutil.rmtree(ex, ignore_errors=True)
        if not keep_zips:
            zpath.unlink(missing_ok=True)


def prepare(out="data/LoveDA", raw=None, synthetic=False, download=False, splits=SPLITS) -> dict:
    spec, out = get_spec("loveda"), Path(out)

    if synthetic:
        spec.make_synthetic(out)
    elif download:
        download_loveda(out, splits)
    elif raw:
        for split in SPLITS:
            z = Path(raw) / f"{split}.zip"
            if z.exists():
                print(f"extracting {z} -> {out}")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
            else:
                print(f"warning: {z} not found, skipping")

    manifest, problems = {}, []
    for split in splits:
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
    ap.add_argument("--download", action="store_true", help="download the official zips from Zenodo (~4.0/2.4/3.1 GB)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)
    prepare(args.out, args.raw, args.synthetic, args.download, tuple(args.splits))


if __name__ == "__main__":
    main()
