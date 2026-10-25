"""Train one (model, fraction, seed) run and write metrics.json.

    python -m dl_research.train --config configs/base.yaml configs/unet_r50.yaml --set fraction=0.25 seed=1

Resumable: re-run the same command and training continues from checkpoints/last.pt (saved at every
validation epoch). A finished run (metrics.json exists) is skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dl_research.config import Config, dump_config, load_config, read_config
from dl_research.data import EpochSampler, SegDataset, get_spec, select_fraction
from dl_research.efficiency import measure_efficiency
from dl_research.evaluate import evaluate_split, make_eval_loader, prepare_data_root, summarize
from dl_research.models import build_model, count_params
from dl_research.utils import env_info, pick_device, read_json, seed_everything, sync, write_json


def resolve_lr(cfg: Config) -> float:
    if cfg.lr is not None:
        return cfg.lr
    try:
        return float(read_json(cfg.tuned_lr_file)[cfg.model]["lr"])
    except (FileNotFoundError, KeyError):
        raise SystemExit(
            f"No learning rate for {cfg.model}: set lr=... or run `python -m dl_research.tune_lr` "
            f"to create {cfg.tuned_lr_file}."
        )


def masked_ce(logits, target, ignore_index):
    """Mean CE over valid pixels; a batch with no valid pixel yields 0 instead of NaN."""
    loss = F.cross_entropy(logits.float(), target, ignore_index=ignore_index, reduction="sum")
    return loss / (target != ignore_index).sum().clamp(min=1)


def lr_lambda(total_steps: int, warmup_frac: float):
    warm = max(1, int(warmup_frac * total_steps))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    return f


def eval_epochs(total: int, num_evals: int) -> set[int]:
    """1-indexed epochs after which validation runs (always includes the last)."""
    return {max(1, round(total * k / num_evals)) for k in range(1, num_evals + 1)} | {total}


def run(cfg: Config, force: bool = False) -> dict:
    run_dir = cfg.run_dir
    if (run_dir / "metrics.json").exists() and not force:
        print(f"[skip] {cfg.run_name}: already complete")
        return read_json(run_dir / "metrics.json")
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = replace(cfg, lr=resolve_lr(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.yaml").exists():
        if read_config(run_dir / "config.yaml") != cfg:
            raise SystemExit(f"{run_dir}/config.yaml differs from the requested config; use --force to restart")
    else:
        dump_config(cfg, run_dir / "config.yaml")

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    spec = get_spec(cfg.dataset)
    root = prepare_data_root(cfg, spec)
    train_samples = select_fraction(spec.scan(root, "Train", True), cfg.fraction, cfg.seed)
    val_samples = spec.scan(root, "Val", True)
    write_json(run_dir / "train_subset.json", [s.id for s in train_samples])

    train_ds = SegDataset(train_samples, spec, True, cfg.crop_size, cfg.crops_per_image, cfg.seed)
    steps_per_epoch = len(train_ds) // cfg.batch_size
    if steps_per_epoch < 1:
        raise SystemExit(f"{len(train_ds)} training crops < batch_size {cfg.batch_size}")
    E = cfg.total_epochs
    total_steps = E * steps_per_epoch

    model = build_model(cfg, spec.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, cfg.warmup_frac))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best_miou, best_epoch, train_time, history = 0, -1.0, -1, 0.0, []
    last_path, best_path = run_dir / "checkpoints" / "last.pt", run_dir / "checkpoints" / "best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best_miou, best_epoch = ck["epoch"], ck["best_miou"], ck["best_epoch"]
        train_time, history = ck["train_time"], ck["history"]
        print(f"[resume] {cfg.run_name} from epoch {start}/{E}")
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=True,
        sampler=EpochSampler(len(train_ds), cfg.batch_size, cfg.seed, start, E),
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda",
    )
    val_loader = make_eval_loader(val_samples, spec, cfg)
    evals = eval_epochs(E, cfg.num_evals)
    it = iter(loader)
    wall0 = time.perf_counter()

    for epoch in range(start, E):
        model.train()
        t0, losses = time.perf_counter(), []
        for _ in range(steps_per_epoch):
            b = next(it)
            x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = masked_ce(logits, y, cfg.ignore_index)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.detach())
        sync(device)
        dt = time.perf_counter() - t0  # training time excludes validation; includes data loading
        train_time += dt
        rec = {"epoch": epoch + 1, "train_loss": torch.stack(losses).mean().item(),
               "lr": sched.get_last_lr()[0], "epoch_time_s": dt}
        if epoch + 1 in evals:
            cms = evaluate_split(model, val_loader, spec, device, cfg.ignore_index, amp=cfg.eval_amp)
            rec["val_miou"] = cms["overall"].compute()["miou"]
            if rec["val_miou"] > best_miou:
                best_miou, best_epoch = rec["val_miou"], epoch + 1
                torch.save({"model": model.state_dict(), "epoch": best_epoch, "miou": best_miou}, best_path)
        history.append(rec)
        print(f"[{cfg.run_name}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if epoch + 1 in evals:
            local_ck = Path(tempfile.gettempdir()) / f"{run_dir.name}_last.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_miou": best_miou,
                        "best_epoch": best_epoch, "train_time": train_time, "history": history}, local_ck)
            # Stage locally, then copy: a stalled Drive write must not leave a truncated last.pt.
            shutil.copyfile(local_ck, last_path.with_suffix(".tmp")); last_path.with_suffix(".tmp").replace(last_path)
    (run_dir / "log.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history))

    # Final evaluation of the best (by validation mIoU) checkpoint, with domain breakdown + sample predictions.
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    cms = evaluate_split(model, val_loader, spec, device, cfg.ignore_index,
                         save_dir=run_dir / "predictions" / "val_best", max_save=cfg.predictions_max_images,
                         amp=cfg.eval_amp)
    bench_size = cfg.crop_size
    metrics = {
        "run": cfg.run_name,
        "complete": True,
        "config": json.loads(json.dumps(cfg.__dict__)),
        "env": env_info(device),
        "model_stats": count_params(model),
        "efficiency": measure_efficiency(model, device, bench_size, cfg.bench_batch_size,
                                         cfg.bench_warmup, cfg.bench_iters),
        "training": {
            "epochs": E, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "n_train_images": len(train_samples), "n_val_images": len(val_samples),
            "train_time_s": train_time, "wall_time_this_session_s": time.perf_counter() - wall0,
            "best_epoch": best_epoch, "final_epoch_val_miou": history[-1].get("val_miou"),
            "train_batch_size": cfg.batch_size, "train_input_size": cfg.crop_size,
        },
        "val": summarize(cms, spec),
    }
    write_json(run_dir / "metrics.json", metrics)
    if cfg.prune_last_checkpoint:  # best.pt is kept; the run is complete so last.pt is no longer needed for resuming
        last_path.unlink(missing_ok=True)
    print(f"[done] {cfg.run_name}: val mIoU {metrics['val']['overall']['miou']:.4f} (best epoch {best_epoch})")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="YAML files, merged left to right")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--force", action="store_true", help="delete an existing run directory and start over")
    args = ap.parse_args(argv)
    run(load_config(args.config, args.set), force=args.force)


if __name__ == "__main__":
    main()
