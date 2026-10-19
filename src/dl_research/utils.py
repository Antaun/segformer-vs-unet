from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Best effort: cuDNN determinism; some ops (e.g. bilinear upsampling backward) stay nondeterministic on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def env_info(device: torch.device) -> dict:
    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        dev_name = "Apple MPS (" + (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()) + ")"
    else:
        dev_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or platform.machine()
    return {
        "device": device.type,
        "device_name": dev_name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "segmentation_models_pytorch": _pkg("segmentation-models-pytorch"),
        "transformers": _pkg("transformers"),
        "torchvision": _pkg("torchvision"),
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
    }


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())
