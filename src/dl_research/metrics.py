"""Confusion-matrix metrics. Pixels whose target is the ignore index never enter the matrix."""
from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.n, self.ignore = num_classes, ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows: target, cols: prediction

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.reshape(-1).long(), target.reshape(-1).long()
        valid = (target != self.ignore) & (target >= 0) & (target < self.n)
        idx = target[valid] * self.n + pred[valid].clamp(0, self.n - 1)
        self.mat += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu().numpy()

    def merge(self, other: "ConfusionMatrix") -> None:
        self.mat += other.mat

    def compute(self) -> dict:
        tp = np.diag(self.mat).astype(np.float64)
        fp, fn = self.mat.sum(0) - tp, self.mat.sum(1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)  # NaN: class absent from GT and preds
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        total = self.mat.sum()
        return {
            "iou": iou,
            "f1": f1,
            "miou": float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float("nan"),
            "mf1": float(np.nanmean(f1)) if not np.all(np.isnan(f1)) else float("nan"),
            "pixel_acc": float(tp.sum() / total) if total else float("nan"),
            "n_pixels": int(total),
        }

    def summary(self, class_names: tuple[str, ...]) -> dict:
        """JSON-friendly version of `compute` (NaN -> None)."""
        r = self.compute()
        nn = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
        return {
            "miou": nn(r["miou"]),
            "mf1": nn(r["mf1"]),
            "pixel_acc": nn(r["pixel_acc"]),
            "n_pixels": r["n_pixels"],
            "iou": {c: nn(v) for c, v in zip(class_names, r["iou"])},
            "f1": {c: nn(v) for c, v in zip(class_names, r["f1"])},
        }
