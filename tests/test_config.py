import pytest

from dl_research.config import load_config


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_merge_order_and_overrides(tmp_path):
    a = write(tmp_path, "a.yaml", "batch_size: 4\nfraction: 0.5\nlr_grids: {unet_r50: [1.0e-4]}\n")
    b = write(tmp_path, "b.yaml", "batch_size: 16\nmodel: segformer_b2\n")
    cfg = load_config([a, b], ["seed=3", "lr=1e-3", "amp=false"])
    assert (cfg.batch_size, cfg.fraction, cfg.model, cfg.seed) == (16, 0.5, "segformer_b2", 3)
    assert cfg.lr == pytest.approx(1e-3) and cfg.amp is False
    assert cfg.lr_grids == {"unet_r50": [1e-4]}


@pytest.mark.parametrize("bad", ["fraction=0", "fraction=1.5", "model=resnet", "crop_size=500", "epochs=0", "optimizer=sgd"])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        load_config([], [bad])


def test_unknown_key_and_malformed_override():
    with pytest.raises(ValueError, match="unknown"):
        load_config([], ["learning_rate=1"])
    with pytest.raises(ValueError, match="key=value"):
        load_config([], ["seed"])


def test_epoch_scaling_and_run_name():
    assert load_config([], ["epochs=30", "fraction=0.1"]).total_epochs == 300
    assert load_config([], ["epochs=30", "fraction=0.25"]).total_epochs == 120
    assert load_config([], ["epochs=30", "fraction=1"]).total_epochs == 30
    assert load_config([], ["epochs=30", "fraction=0.1", "scale_epochs_by_fraction=false"]).total_epochs == 30
    assert load_config([], ["model=segformer_b2", "fraction=0.25", "seed=2"]).run_name == "segformer_b2_f0.25_s2"


def test_shipped_configs_parse():
    for model in ("unet_r50", "segformer_b2"):
        cfg = load_config(["configs/base.yaml", f"configs/{model}.yaml"])
        assert cfg.model == model and cfg.pretrained and cfg.crop_size == 512
    assert load_config(["configs/base.yaml", "configs/smoke.yaml"]).smoke
