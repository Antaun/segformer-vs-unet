#!/usr/bin/env bash
# Full study: LR tuning (validation only), then {unet_r50, segformer_b2} x {1.0, 0.25, 0.1} x seeds, then tables/plots.
# Re-running resumes/skips finished runs. Override with e.g. SEEDS="0 1 2 3 4" FRACTIONS="1.0 0.25 0.1".
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
FRACTIONS=${FRACTIONS:-"1.0 0.25 0.1"}
MODELS=${MODELS:-"unet_r50 segformer_b2"}

python -m dl_research.tune_lr --config configs/base.yaml --models $MODELS
for seed in $SEEDS; do
  for frac in $FRACTIONS; do
    for m in $MODELS; do
      python -m dl_research.train --config configs/base.yaml configs/$m.yaml --set seed=$seed fraction=$frac
    done
  done
done
python -m dl_research.aggregate --exp-dir experiments --out results
