import numpy as np
import pytest
import torch

from dl_research.metrics import ConfusionMatrix


def test_known_values():
    target = torch.tensor([0, 0, 0, 1, 1, 2])
    pred = torch.tensor([0, 0, 1, 1, 1, 0])
    cm = ConfusionMatrix(3)
    cm.update(pred, target)
    r = cm.compute()
    # class0: tp2 fp1 fn1 -> IoU 2/4; class1: tp2 fp1 fn0 -> 2/3; class2: tp0 fn1 -> 0
    assert r["iou"] == pytest.approx([0.5, 2 / 3, 0.0])
    assert r["miou"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert r["pixel_acc"] == pytest.approx(4 / 6)
    assert r["f1"] == pytest.approx([4 / 6, 4 / 5, 0.0])  # 2tp / (2tp + fp + fn)
    assert r["n_pixels"] == 6


def test_ignore_index_excluded():
    cm = ConfusionMatrix(2, ignore_index=255)
    cm.update(torch.tensor([0, 1, 1, 0]), torch.tensor([0, 255, 1, 255]))
    assert cm.mat.sum() == 2 and cm.compute()["pixel_acc"] == 1.0


def test_absent_class_is_nan_and_skipped_in_mean():
    cm = ConfusionMatrix(3)
    cm.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    r = cm.compute()
    assert np.isnan(r["iou"][2]) and r["miou"] == 1.0
    assert cm.summary(("a", "b", "c"))["iou"]["c"] is None


def test_all_ignored_gives_nan_not_crash():
    cm = ConfusionMatrix(2)
    cm.update(torch.tensor([0, 1]), torch.tensor([255, 255]))
    assert np.isnan(cm.compute()["miou"]) and np.isnan(cm.compute()["pixel_acc"])


def test_accumulation_and_merge_match_single_pass():
    g = torch.Generator().manual_seed(0)
    p, t = torch.randint(0, 4, (2, 500), generator=g)
    whole = ConfusionMatrix(4); whole.update(p, t)
    a, b = ConfusionMatrix(4), ConfusionMatrix(4)
    a.update(p[:200], t[:200]); b.update(p[200:], t[200:]); a.merge(b)
    assert (a.mat == whole.mat).all()
    assert a.compute()["miou"] == pytest.approx(whole.compute()["miou"])
