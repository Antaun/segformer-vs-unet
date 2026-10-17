"""Experiment configuration: YAML files are merged left to right, then `key=value` overrides."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

MODELS = ("unet_r50", "segformer_b2")


@dataclass
class Config:
    model: str = "unet_r50"
    dataset: str = "loveda"
    data_root: str = "data/LoveDA"
    out_dir: str = "experiments"
    seed: int = 0
    fraction: float = 1.0
    crop_size: int = 512
    crops_per_image: int = 1
    batch_size: int = 8
    eval_batch_size: int = 2
    num_workers: int = 4
    epochs: int = 30
    scale_epochs_by_fraction: bool = True
    num_evals: int = 10
    optimizer: str = "adamw"
    lr: float | None = None
    tuned_lr_file: str = "configs/tuned_lr.json"
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    amp: bool = True
    pretrained: bool = True
    encoder: str = "resnet50"
    segformer_checkpoint: str = "nvidia/mit-b2"
    ignore_index: int = 255
    predictions_max_images: int = 20
    tune_fraction: float = 1.0
    tune_epochs: int = 10
    tune_seed: int = 0
    lr_grids: dict = field(default_factory=dict)
    bench_warmup: int = 10
    bench_iters: int = 50
    bench_batch_size: int = 1
    smoke: bool = False
    device: str = "auto"

    def validate(self) -> "Config":
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.crop_size % 32:
            raise ValueError(f"crop_size must be a multiple of 32, got {self.crop_size}")
        if self.optimizer != "adamw":
            raise ValueError("only optimizer: adamw is implemented")
        for name in ("epochs", "batch_size", "eval_batch_size", "crops_per_image", "num_evals"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        return self

    @property
    def total_epochs(self) -> int:
        """Epochs actually run. With scaling, every fraction gets (about) the same number of steps."""
        if self.scale_epochs_by_fraction:
            return max(1, round(self.epochs / self.fraction))
        return self.epochs

    @property
    def run_name(self) -> str:
        return f"{self.model}_f{self.fraction:g}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_config(raw: dict[str, Any]) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    # YAML 1.1 reads `1e-3` as a string; coerce float-typed fields explicitly.
    for name in ("lr", "fraction", "weight_decay", "warmup_frac", "tune_fraction"):
        if isinstance(raw.get(name), (str, int)) and not isinstance(raw.get(name), bool):
            raw[name] = float(raw[name])
    return Config(**raw).validate()


def load_config(paths: list[str] | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    for p in paths or []:
        raw = _deep_merge(raw, yaml.safe_load(Path(p).read_text()) or {})
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        raw[key.strip()] = yaml.safe_load(value)
    return build_config(raw)


def dump_config(cfg: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))


def read_config(path: Path) -> Config:
    return build_config(yaml.safe_load(Path(path).read_text()))
