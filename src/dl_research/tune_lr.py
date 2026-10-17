"""Learning-rate tuning on the VALIDATION split only.

For each model, train one short run (tune_epochs, tune_fraction, tune_seed) per candidate in `lr_grids`, and keep
the LR whose best-checkpoint validation mIoU is highest. The choice is written to configs/tuned_lr.json and read
by `train` when `lr: null`.

    python -m dl_research.tune_lr --config configs/base.yaml [--models unet_r50 segformer_b2] [--set key=value ...]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dl_research.config import Config, load_config
from dl_research.train import run
from dl_research.utils import read_json, write_json


def tune(base: Config, models=("unet_r50", "segformer_b2")) -> dict:
    out_file = Path(base.tuned_lr_file)
    tuned = read_json(out_file) if out_file.exists() else {}
    for model in models:
        scores = {}
        for lr in base.lr_grids[model]:
            cfg = replace(
                base, model=model, lr=float(lr), fraction=base.tune_fraction, epochs=base.tune_epochs,
                scale_epochs_by_fraction=False, seed=base.tune_seed,
                out_dir=str(Path(base.out_dir) / "tune" / f"lr{lr:g}"),
            ).validate()
            scores[lr] = run(cfg)["val"]["overall"]["miou"]
        best = max(scores, key=scores.get)
        tuned[model] = {
            "lr": best,
            "val_miou": scores[best],
            "grid_val_miou": {f"{k:g}": v for k, v in scores.items()},
            "method": "grid search; one run per LR; best-checkpoint validation mIoU; validation split only",
            "tune_epochs": base.tune_epochs, "tune_fraction": base.tune_fraction, "tune_seed": base.tune_seed,
        }
        write_json(out_file, tuned)
        print(f"[tune] {model}: lr={best:g} (val mIoU {scores[best]:.4f}) -> {out_file}")
    return tuned


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--models", nargs="+", default=["unet_r50", "segformer_b2"])
    args = ap.parse_args(argv)
    tune(load_config(args.config, args.set), args.models)


if __name__ == "__main__":
    main()
