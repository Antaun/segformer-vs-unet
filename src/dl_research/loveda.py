"""LoveDA spec: layout scanning, label mapping, and a synthetic generator for smoke tests.

Official layout (after extracting Train.zip / Val.zip / Test.zip into `data/LoveDA/`):

    <root>/{Train,Val,Test}/{Urban,Rural}/images_png/*.png
    <root>/{Train,Val}/{Urban,Rural}/masks_png/*.png      # Test masks are not public

The zips unpack to `Train/Train/...`; both nestings are accepted.

Raw mask values: 0 = no-data, 1 = background, 2 = building, 3 = road, 4 = water, 5 = barren, 6 = forest,
7 = agriculture. Training ids are raw - 1 (background is class 0, so 7 classes) and raw 0 (plus any other
unexpected value) becomes the ignore index 255.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dl_research.data import IGNORE_INDEX, DatasetSpec, Sample, register

CLASS_NAMES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
DOMAINS = ("urban", "rural")  # lower-case ids; directories are Urban / Rural
SPLITS = ("Train", "Val", "Test")

_decode = np.full(256, IGNORE_INDEX, dtype=np.uint8)
_decode[1:8] = np.arange(7)
_encode = np.arange(1, 8, dtype=np.uint8)


def scan_loveda(root: str | Path, split: str, require_labels: bool = True) -> list[Sample]:
    base = Path(root) / split
    if (base / split).is_dir():
        base = base / split
    samples: list[Sample] = []
    for dom in DOMAINS:
        img_dir, mask_dir = base / dom.capitalize() / "images_png", base / dom.capitalize() / "masks_png"
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"{img_dir} not found. Expected <root>/{split}/{dom.capitalize()}/images_png; "
                "see README 'Dataset' or run `python -m dl_research.prepare_dataset`."
            )
        for p in sorted(img_dir.glob("*.png")):
            m = mask_dir / p.name
            if not m.exists():
                if require_labels:
                    raise FileNotFoundError(f"missing mask for {p} (expected {m})")
                m = None
            samples.append(Sample(str(p), str(m) if m else None, dom, f"{dom}_{p.stem}"))
    if not samples:
        raise FileNotFoundError(f"no .png images found under {base}")
    return samples


def make_synthetic(root: str | Path, counts: dict[str, int] | None = None, size: int = 128, seed: int = 0) -> None:
    """Write a tiny LoveDA-shaped dataset (random images, blocky masks incl. raw 0 pixels). Test has no masks."""
    counts = counts or {"Train": 4, "Val": 4, "Test": 2}
    rng = np.random.default_rng(seed)
    for split, n in counts.items():
        for dom in DOMAINS:
            d = Path(root) / split / dom.capitalize()
            (d / "images_png").mkdir(parents=True, exist_ok=True)
            if split != "Test":
                (d / "masks_png").mkdir(parents=True, exist_ok=True)
            for i in range(n):
                blocks = rng.integers(0, 8, size=(4, 4), dtype=np.uint8)  # raw values incl. 0 (no-data)
                mask = np.kron(blocks, np.ones((size // 4, size // 4), dtype=np.uint8))
                img = (rng.random((size, size, 3)) * 80 + mask[..., None] * 20).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(d / "images_png" / f"{i}.png")
                if split != "Test":
                    Image.fromarray(mask).save(d / "masks_png" / f"{i}.png")


LOVEDA = register(
    DatasetSpec(
        name="loveda",
        class_names=CLASS_NAMES,
        domains=DOMAINS,
        decode_lut=_decode,
        encode_lut=_encode,
        scan=scan_loveda,
        make_synthetic=make_synthetic,
    )
)
