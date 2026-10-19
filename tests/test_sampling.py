import numpy as np

from dl_research.data import EpochSampler, Sample, SegDataset, get_spec, select_fraction
from dl_research.loveda import make_synthetic, scan_loveda


def fake(n_urban=20, n_rural=30):
    return [Sample(f"i{d}{i}", f"m{d}{i}", d, f"{d}_{i}") for d, n in (("urban", n_urban), ("rural", n_rural)) for i in range(n)]


def ids(xs):
    return [s.id for s in xs]


def test_deterministic_and_order_independent():
    s = fake()
    assert ids(select_fraction(s, 0.25, 7)) == ids(select_fraction(s, 0.25, 7))
    assert ids(select_fraction(s, 0.25, 7)) == ids(select_fraction(s[::-1], 0.25, 7))


def test_seed_changes_subset():
    s = fake()
    assert ids(select_fraction(s, 0.25, 0)) != ids(select_fraction(s, 0.25, 1))


def test_stratified_by_domain_and_sizes():
    sub = select_fraction(fake(20, 30), 0.25, 0)
    assert sum(x.domain == "urban" for x in sub) == 5 and sum(x.domain == "rural" for x in sub) == round(7.5)
    assert len(set(ids(sub))) == len(sub)


def test_nested_subsets_and_full():
    s = fake()
    f10, f25, f100 = (set(ids(select_fraction(s, f, 3))) for f in (0.1, 0.25, 1.0))
    assert f10 <= f25 <= f100 and f100 == set(ids(s))


def test_tiny_fraction_keeps_at_least_one_per_domain():
    sub = select_fraction(fake(3, 3), 0.01, 0)
    assert sorted(x.domain for x in sub) == ["rural", "urban"]


def test_epoch_sampler_reproducible_resume_and_batches():
    full = list(EpochSampler(10, 4, seed=1, start_epoch=0, end_epoch=3))
    assert len(full) == 3 * 8 and len(EpochSampler(10, 4, 1, 0, 3)) == 24
    resumed = list(EpochSampler(10, 4, seed=1, start_epoch=2, end_epoch=3))
    assert resumed == full[16:]
    assert {i // 10 for i in full} == {0, 1, 2}


def test_augmentation_is_pure_function_of_seed_epoch_index(tmp_path):
    spec = get_spec("loveda")
    make_synthetic(tmp_path, {"Train": 2}, size=96)
    samples = scan_loveda(tmp_path, "Train")
    ds = SegDataset(samples, spec, True, crop_size=64, seed=5)
    a, b = ds[1], ds[1]
    assert (a["image"] == b["image"]).all() and (a["mask"] == b["mask"]).all()
    assert a["image"].shape == (3, 64, 64) and a["mask"].shape == (64, 64)
    later_epoch = ds[len(ds) + 1]  # same image, next epoch -> (usually) different crop/flip
    assert not (later_epoch["image"] == a["image"]).all() or not (later_epoch["mask"] == a["mask"]).all()
    assert SegDataset(samples, spec, True, 64, seed=6)[1]["image"].shape == a["image"].shape


def test_small_image_is_padded_with_ignore(tmp_path):
    spec = get_spec("loveda")
    make_synthetic(tmp_path, {"Train": 1}, size=32)
    item = SegDataset(scan_loveda(tmp_path, "Train"), spec, True, crop_size=64, seed=0)[0]
    assert item["mask"].shape == (64, 64) and (item["mask"] == 255).any()
