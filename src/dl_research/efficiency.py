"""Model cost, measured the same way for every model: fp32, eval mode, no_grad, random input."""
from __future__ import annotations

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from dl_research.utils import sync


@torch.no_grad()
def measure_efficiency(model, device, size: int, batch_size: int = 1, warmup: int = 10, iters: int = 50) -> dict:
    model = model.to(device).eval()
    # FLOPs: torch's FlopCounterMode counts conv/matmul/attention ops (1 MAC = 2 FLOPs) for one 1 x 3 x size x size image.
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, size, size, device=device))
    x = torch.randn(batch_size, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    return {
        "gflops_per_image": fc.get_total_flops() / 1e9,
        "inference_ms_per_batch": dt * 1e3,
        "inference_ms_per_image": dt * 1e3 / batch_size,
        "inference_images_per_s": batch_size / dt,
        "bench_input_size": size,
        "bench_batch_size": batch_size,
        "bench_warmup": warmup,
        "bench_iters": iters,
        "bench_precision": "fp32",
    }
