"""Dataset-agnostic pieces: sample records, the dataset registry, deterministic subsets, SegDataset.

A new dataset (e.g. DeepGlobe) only needs a `DatasetSpec` registered in `DATASETS`; nothing here or in
train/evaluate changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE_INDEX = 255


@dataclass(frozen=True)
class Sample:
    image: str
    mask: str | None  # None when labels are unavailable (LoveDA test)
    domain: str
    id: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    class_names: tuple[str, ...]
    domains: tuple[str, ...]
    decode_lut: np.ndarray  # raw mask value (0..255) -> train id (0..C-1) or IGNORE_INDEX
    encode_lut: np.ndarray  # train id -> raw mask value, for writing predictions
    scan: Callable[[Path, str, bool], list[Sample]]  # (root, split, require_labels) -> samples
    make_synthetic: Callable[..., None] | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


DATASETS: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    DATASETS[spec.name] = spec
    return spec


def get_spec(name: str) -> DatasetSpec:
    import dl_research.loveda  # noqa: F401  (registers on import)

    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name]


def select_fraction(samples: list[Sample], fraction: float, seed: int) -> list[Sample]:
    """Deterministic label-fraction subset of training IMAGES, stratified by domain.

    Each domain is shuffled with a seeded permutation and its first `round(fraction * n)` images are kept, so
    (a) every domain keeps its proportion, (b) the result depends only on (samples, fraction, seed), and
    (c) subsets are nested: the 10% subset is contained in the 25% subset, which is contained in the 100% set.
    Class balance is NOT controlled.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    out: list[Sample] = []
    for di, dom in enumerate(sorted({s.domain for s in samples})):
        group = sorted((s for s in samples if s.domain == dom), key=lambda s: s.id)
        n = len(group) if fraction == 1 else max(1, round(fraction * len(group)))
        perm = np.random.default_rng([seed, di]).permutation(len(group))
        out += [group[i] for i in sorted(perm[:n])]
    return out


def decode_mask(raw: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    """Raw mask -> train ids; invalid/no-data raw values become IGNORE_INDEX."""
    if raw.ndim == 3:
        raw = raw[..., 0]
    return spec.decode_lut[raw]


def _augment(img: np.ndarray, mask: np.ndarray, crop: int, rng: np.random.Generator):
    h, w = mask.shape
    if h < crop or w < crop:  # pad: image with 0, mask with ignore
        ph, pw = max(0, crop - h), max(0, crop - w)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
        mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=IGNORE_INDEX)
        h, w = mask.shape
    y, x = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
    img, mask = img[y : y + crop, x : x + crop], mask[y : y + crop, x : x + crop]
    if rng.random() < 0.5:  # horizontal flip
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:  # vertical flip
        img, mask = img[::-1], mask[::-1]
    k = int(rng.integers(0, 4))  # rotation by k * 90 degrees (lossless for label maps)
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SegDataset(Dataset):
    """Training mode: index = epoch * len + i, so augmentation is a pure function of (seed, epoch, i)
    regardless of worker count or resume point (see `EpochSampler`). Eval mode: full images, plain indices."""

    def __init__(self, samples, spec, train, crop_size=512, crops_per_image=1, seed=0):
        self.samples, self.spec, self.train = list(samples), spec, train
        self.crop_size, self.crops_per_image, self.seed = crop_size, crops_per_image, seed

    def __len__(self) -> int:
        return len(self.samples) * (self.crops_per_image if self.train else 1)

    def __getitem__(self, index: int) -> dict:
        n = len(self)
        epoch, i = divmod(index, n) if self.train else (0, index)
        s = self.samples[i % len(self.samples)]
        img = np.asarray(Image.open(s.image).convert("RGB"))
        if s.mask is not None:
            mask = decode_mask(np.asarray(Image.open(s.mask)), self.spec)
        else:
            mask = np.full(img.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        if self.train:
            img, mask = _augment(img, mask, self.crop_size, np.random.default_rng([self.seed, epoch, i]))
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(x).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "domain": self.spec.domains.index(s.domain),
            "index": i % len(self.samples),
        }


class EpochSampler(Sampler):
    """One sampler spanning epochs [start, end): lets a single persistent DataLoader serve the whole run and
    makes resuming exact. Each epoch is a seeded permutation truncated to a whole number of batches."""

    def __init__(self, n: int, batch_size: int, seed: int, start_epoch: int, end_epoch: int):
        self.n, self.bs, self.seed, self.start, self.end = n, batch_size, seed, start_epoch, end_epoch

    def __len__(self) -> int:
        return (self.end - self.start) * (self.n // self.bs * self.bs)

    def __iter__(self):
        for e in range(self.start, self.end):
            perm = np.random.default_rng([self.seed, e]).permutation(self.n)[: self.n // self.bs * self.bs]
            yield from (e * self.n + int(i) for i in perm)
