#!/usr/bin/env bash
# Tiny end-to-end check on synthetic data (CPU, no downloads, ~1 minute).
set -euo pipefail
for m in unet_r50 segformer_b2; do
  python -m dl_research.train --config configs/base.yaml configs/smoke.yaml --set model=$m tuned_lr_file=none
done
python -m dl_research.aggregate --exp-dir experiments/smoke --out experiments/smoke/results
