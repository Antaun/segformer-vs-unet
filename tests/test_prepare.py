import hashlib
import shutil

import pytest

from dl_research import prepare_dataset as pd_
from dl_research.config import load_config
from dl_research.loveda import make_synthetic, scan_loveda


@pytest.mark.parametrize("nesting", ["", "Train", "wrapper/Train"])
def test_organize_handles_any_nesting(tmp_path, nesting):
    raw = tmp_path / "raw" / nesting
    make_synthetic(raw, {"Train": 2}, size=32)  # -> raw[/nesting]/Train/{Urban,Rural}/...
    # Kaggle-style double nesting: raw/Train/Train/Urban
    out = tmp_path / "out"
    found = pd_.organize(tmp_path / "raw", out)
    assert set(found) == {"Train"} and len(scan_loveda(out, "Train")) == 4


def test_organize_does_not_overwrite_existing(tmp_path, capsys):
    make_synthetic(tmp_path / "raw", {"Train": 1}, size=32)
    make_synthetic(tmp_path / "out", {"Train": 3}, size=32)
    pd_.organize(tmp_path / "raw", tmp_path / "out")
    assert len(scan_loveda(tmp_path / "out", "Train")) == 6  # untouched (3 per domain)
    assert "skip" in capsys.readouterr().out


def test_download_file_skips_valid_existing_file(tmp_path):
    f = tmp_path / "x.zip"
    f.write_bytes(b"hello")
    pd_.download_file("http://invalid.invalid/never-called", f, hashlib.md5(b"hello").hexdigest())  # would raise if curl ran


def test_download_file_retries_interrupted_transfers(tmp_path, monkeypatch):
    f = tmp_path / "x.zip"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        f.write_bytes(b"hello")

    monkeypatch.setattr(pd_.subprocess, "run", fake_run)
    pd_.download_file("https://example.invalid/x.zip", f, hashlib.md5(b"hello").hexdigest())

    assert commands[0][commands[0].index("--retry") + 1] == "5"
    assert "--retry-all-errors" in commands[0]
    assert commands[0][commands[0].index("-C") + 1] == "-"


def test_download_file_rejects_checksum_mismatch(tmp_path, monkeypatch):
    f = tmp_path / "x.zip"
    monkeypatch.setattr(pd_.subprocess, "run", lambda *a, **k: f.write_bytes(b"corrupt"))
    with pytest.raises(RuntimeError, match="MD5"):
        pd_.download_file("http://example.invalid", f, "0" * 32)


def test_zenodo_table_is_complete():
    assert set(pd_.ZENODO) == {"Train", "Val", "Test"} and all(len(v[1]) == 32 for v in pd_.ZENODO.values())


def test_new_options_default_off_and_parse():
    cfg = load_config([], [])
    assert cfg.eval_amp is False and cfg.prune_last_checkpoint is False
    assert load_config([], ["eval_amp=true", "prune_last_checkpoint=true"]).prune_last_checkpoint is True
