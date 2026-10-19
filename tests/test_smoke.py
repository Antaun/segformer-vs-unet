"""End-to-end on synthetic data: no dataset, no downloads, CPU only."""
import json

import pytest

from dl_research import aggregate, evaluate, train
from dl_research.config import load_config

SMOKE = ["configs/base.yaml", "configs/smoke.yaml"]


def smoke_cfg(tmp_path, model, **kw):
    over = [f"model={model}", f"out_dir={tmp_path}", "tuned_lr_file=none"] + [f"{k}={v}" for k, v in kw.items()]
    return load_config(SMOKE, over)


@pytest.mark.parametrize("model", ["unet_r50", "segformer_b2"])
def test_train_evaluate_resume(tmp_path, model):
    cfg = smoke_cfg(tmp_path, model)
    m = train.run(cfg)
    run_dir = tmp_path / cfg.run_name
    for f in ("config.yaml", "log.jsonl", "metrics.json", "train_subset.json", "checkpoints/best.pt", "checkpoints/last.pt"):
        assert (run_dir / f).exists(), f
    assert list((run_dir / "predictions" / "val_best").glob("*.png"))
    assert set(m["val"]) == {"overall", "urban", "rural"}
    assert 0 <= m["val"]["overall"]["miou"] <= 1 and 0 <= m["val"]["overall"]["pixel_acc"] <= 1
    assert set(m["val"]["overall"]["iou"]) == {"background", "building", "road", "water", "barren", "forest", "agriculture"}
    assert m["model_stats"]["params_total"] > 0 and m["efficiency"]["inference_ms_per_image"] > 0
    assert m["env"]["device"] == "cpu" and m["training"]["train_input_size"] == 64
    json.dumps(m)  # machine readable
    assert train.run(cfg)["run"] == cfg.run_name  # finished run is skipped, not retrained

    evaluate.main(["--run", str(run_dir), "--split", "val"])
    assert (run_dir / "eval_val_best.json").exists()
    evaluate.main(["--run", str(run_dir), "--split", "test"])  # no labels: predictions only
    assert list((run_dir / "predictions" / "test_best").glob("*.png"))


def test_resume_matches_uninterrupted_run(tmp_path):
    full = train.run(smoke_cfg(tmp_path / "a", "unet_r50", epochs=4, num_evals=4))
    cfg_b = smoke_cfg(tmp_path / "b", "unet_r50", epochs=4, num_evals=4)
    train.run(cfg_b)
    (cfg_b.run_dir / "metrics.json").unlink()  # pretend we were killed after the last checkpoint
    again = train.run(cfg_b)
    assert again["val"]["overall"]["miou"] == pytest.approx(full["val"]["overall"]["miou"])


def test_lr_required_when_untuned(tmp_path):
    cfg = load_config(SMOKE, [f"out_dir={tmp_path}", "lr=null", "tuned_lr_file=missing.json"])
    with pytest.raises(SystemExit, match="learning rate"):
        train.run(cfg)


def test_aggregate_over_smoke_runs(tmp_path):
    for model in ("unet_r50", "segformer_b2"):
        for frac in (1.0, 0.5):
            train.run(smoke_cfg(tmp_path, model, fraction=frac))
    df = aggregate.load_runs(tmp_path)
    assert len(df) == 4
    aggregate.make_tables(df, tmp_path / "out")
    aggregate.make_plots(df, tmp_path / "out")
    assert "Paired gap" in (tmp_path / "out" / "summary.md").read_text()
    assert (tmp_path / "out" / "figures" / "miou_vs_fraction.png").exists()


def test_prune_last_checkpoint_keeps_best(tmp_path):
    cfg = smoke_cfg(tmp_path, "unet_r50", prune_last_checkpoint="true", eval_amp="true")
    train.run(cfg)
    assert (cfg.run_dir / "checkpoints" / "best.pt").exists()
    assert not (cfg.run_dir / "checkpoints" / "last.pt").exists()
    assert train.run(cfg)["run"] == cfg.run_name  # still skipped as complete without last.pt
