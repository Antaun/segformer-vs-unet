import numpy as np
import pytest
from PIL import Image

from dl_research.data import decode_mask, get_spec
from dl_research.loveda import CLASS_NAMES, make_synthetic, scan_loveda

spec = get_spec("loveda")


def test_decode_lut():
    raw = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 100, 255]], dtype=np.uint8)
    out = decode_mask(raw, spec)
    assert out[0].tolist() == [255, 0, 1, 2]       # no-data -> ignore, background -> 0
    assert out[1].tolist() == [3, 4, 5, 6]
    assert out[2].tolist() == [255, 255, 255, 255]  # invalid values never become a class
    assert spec.num_classes == 7 and CLASS_NAMES[0] == "background"


def test_encode_roundtrip():
    ids = np.arange(7, dtype=np.uint8)
    assert decode_mask(spec.encode_lut[ids], spec).tolist() == ids.tolist()


def test_scan_synthetic_layout(tmp_path):
    make_synthetic(tmp_path, {"Train": 3, "Val": 2, "Test": 2}, size=32)
    train = scan_loveda(tmp_path, "Train")
    assert len(train) == 6 and {s.domain for s in train} == {"urban", "rural"}
    assert all(s.mask for s in train)
    test = scan_loveda(tmp_path, "Test", require_labels=False)
    assert len(test) == 4 and all(s.mask is None for s in test)
    with pytest.raises(FileNotFoundError):
        scan_loveda(tmp_path, "Test", require_labels=True)


def test_scan_accepts_nested_zip_layout(tmp_path):
    make_synthetic(tmp_path / "nested" / "Train", {"Train": 1}, size=32)  # -> nested/Train/Train/Urban/...
    assert len(scan_loveda(tmp_path / "nested", "Train")) == 2


def test_scan_missing_dir_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="images_png"):
        scan_loveda(tmp_path, "Train")


def test_masks_with_nodata_are_ignored_end_to_end(tmp_path):
    from dl_research.data import SegDataset
    make_synthetic(tmp_path, {"Train": 2}, size=32)
    s = scan_loveda(tmp_path, "Train")[0]
    raw = np.asarray(Image.open(s.mask))
    item = SegDataset([s], spec, train=False)[0]
    assert ((item["mask"].numpy() == 255) == (raw == 0)).all()
