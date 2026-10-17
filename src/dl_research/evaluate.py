"""Evaluation on full-resolution images, overall and per domain. Also a CLI to re-evaluate a finished run.

    python -m dl_research.evaluate --run experiments/unet_r50_f1_s0 --split val [--ckpt best] [--max-save 50]

`--split test` only writes predictions: LoveDA test labels are not public, so no metrics are computed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dl_research.config import Config, read_config
from dl_research.data import SegDataset, get_spec
from dl_research.metrics import ConfusionMatrix
from dl_research.models import build_model
from dl_research.utils import pick_device, write_json


def prepare_data_root(cfg: Config, spec) -> str:
    """Real data root, or (smoke mode) a tiny synthetic dataset generated inside the run directory."""
    if not cfg.smoke:
        return cfg.data_root
    root = cfg.run_dir / "_synthetic_data"
    if not root.exists():
        spec.make_synthetic(root, size=2 * cfg.crop_size)
    return str(root)


def make_eval_loader(samples, spec, cfg: Config) -> DataLoader:
    ds = SegDataset(samples, spec, train=False)
    return DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)


@torch.no_grad()
def evaluate_split(model, loader, spec, device, ignore_index=255, labeled=True, save_dir=None, max_save=0):
    """Returns {"overall": CM, "<domain>": CM, ...}. Predictions are written with RAW label values (1..7)."""
    model.eval()
    cms = {k: ConfusionMatrix(spec.num_classes, ignore_index) for k in ("overall", *spec.domains)}
    samples, saved = loader.dataset.samples, 0
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        pred = model(batch["image"].to(device)).argmax(1).cpu()
        for p, y, d, i in zip(pred, batch["mask"], batch["domain"].tolist(), batch["index"].tolist()):
            if labeled:
                cms["overall"].update(p, y)
                cms[spec.domains[d]].update(p, y)
            if save_dir and saved < max_save:
                Image.fromarray(spec.encode_lut[p.numpy()]).save(Path(save_dir) / f"{samples[i].id}.png")
                saved += 1
    return cms


def summarize(cms: dict, spec) -> dict:
    return {k: cm.summary(spec.class_names) for k, cm in cms.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing config.yaml and checkpoints/")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--max-save", type=int, default=None, help="max predictions to write (default: all)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    cfg = read_config(run / "config.yaml")
    spec = get_spec(cfg.dataset)
    device = pick_device(cfg.device)
    model = build_model(cfg, spec.num_classes).to(device)
    state = torch.load(run / "checkpoints" / f"{args.ckpt}.pt", map_location=device)
    model.load_state_dict(state["model"])

    split = "Val" if args.split == "val" else "Test"
    labeled = args.split == "val"
    samples = spec.scan(prepare_data_root(cfg, spec), split, labeled)
    loader = make_eval_loader(samples, spec, cfg)
    max_save = len(samples) if args.max_save is None else args.max_save
    cms = evaluate_split(model, loader, spec, device, cfg.ignore_index, labeled,
                         run / "predictions" / f"{args.split}_{args.ckpt}", max_save)
    if labeled:
        out = {"split": args.split, "ckpt": args.ckpt, "epoch": state.get("epoch"), **summarize(cms, spec)}
        write_json(run / f"eval_{args.split}_{args.ckpt}.json", out)
        print(f"{args.split} mIoU: {out['overall']['miou']:.4f}  pixel acc: {out['overall']['pixel_acc']:.4f}")
    else:
        print(f"wrote {min(max_save, len(samples))} test predictions (no labels -> no metrics)")


if __name__ == "__main__":
    main()
