# segformer-vs-unet

A reproducible comparison of **U-Net with a ResNet-50 encoder** (`segmentation_models_pytorch`) and
**SegFormer-B2 / MiT-B2** (Hugging Face Transformers) for semantic segmentation on **LoveDA** (7 classes incl.
background; urban and rural domains).

> **Status: code only, no experiments have been run yet.** This repository contains no measured results. Everything
> under "Hypotheses" is a plan, not a finding. Results appear only in `results/` after you run the pipeline.

## Research questions
1. Under one shared protocol, how do the two models compare in mIoU?
2. How does the gap change with 100% / 25% / 10% of the training labels?
3. Does the relative performance differ between the urban and rural domains?
4. How do parameters, FLOPs, training time and inference speed relate to segmentation quality?

### Hypotheses (planned, unverified)
These are expectations to be tested, not results: (H1) the pretrained transformer encoder may help more when labels are
scarce; (H2) domain shift between urban and rural may affect the two models differently; (H3) the models differ
substantially in compute cost. Any of these may turn out false.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest                      # 43 tests, ~1 min, no dataset needed
```
Developed with Python 3.14, torch 2.14.1, torchvision 0.29.1, segmentation-models-pytorch 0.5.0, timm 1.0.30,
transformers 5.19.0 (macOS, Apple silicon). Every `metrics.json` records the exact versions and device it ran on.
Pretrained weights (ResNet-50 ImageNet via smp, `nvidia/mit-b2` via the HF hub) are downloaded on first use of a
non-smoke run. **That path has not been exercised in development** (no downloads were made); verify it with one short
real run before launching the study.

## Dataset (LoveDA)
Opt-in download of the official files from Zenodo (record 5706578; MD5-verified; Train 4.0 GB, Val 2.4 GB, Test 3.1 GB —
Test has no labels and is not needed here):
```bash
python -m dl_research.prepare_dataset --out data/LoveDA --download --splits Train Val
```
Or by hand:
Download `Train.zip`, `Val.zip`, `Test.zip` from the official LoveDA release (github.com/Junjue-Wang/LoveDA) and run:
```bash
python -m dl_research.prepare_dataset --raw /path/to/zips --out data/LoveDA   # unzips, verifies, writes manifest.json
```
Expected structure (`Train/Train/...` nesting from the zips is also accepted):
```
data/LoveDA/{Train,Val,Test}/{Urban,Rural}/images_png/*.png
data/LoveDA/{Train,Val}/{Urban,Rural}/masks_png/*.png       

# Test masks are not public
```
- Raw mask values: 0 no-data, 1 background, 2 building, 3 road, 4 water, 5 barren, 6 forest, 7 agriculture.
  Training ids = raw − 1 (7 classes, background = class 0). **Raw 0 and any other value become the ignore index 255**: excluded from the loss and from every metric.
- Domain (urban/rural) comes from the directory, so domain metrics exist for every labeled split.
- `prepare_dataset` warns if image counts differ from the published ones (these counts are recorded in the script
  from memory of the release; check them against your copy).

## Protocol (identical for both models, `configs/base.yaml`)
| Item | Setting |
|---|---|
| Input | random 512×512 crops of the 1024×1024 images, ImageNet mean/std |
| Augmentation | horizontal flip, vertical flip, rotation by k·90° (random k) — lossless for label maps; arbitrary-angle rotation is not implemented |
| Loss / optimizer | cross-entropy (ignore 255), AdamW, weight decay 0.01, 5% linear warm-up then cosine decay |
| Batch size | 8 (fp16 autocast on CUDA only; MPS/CPU use fp32) |
| Epochs | `epochs: 30` base, identical for both models; **scaled by 1/fraction** (30 / 120 / 300) so every fraction gets about the same number of optimizer steps. Set `scale_epochs_by_fraction: false` for a fixed epoch count instead. |
| Seeds | one seed drives init, subset choice and augmentation; default study uses seeds 0, 1, 2 |
| Evaluation | full 1024×1024 validation images, fp32, argmax |

**Label fractions.** Subsets are of whole training *images*, stratified by domain (each domain keeps `round(f·n)` images),
seeded, and *nested* (10% ⊂ 25% ⊂ 100% for a given seed). Class balance is not controlled. Both models see the same subset for a given seed; different seeds draw different subsets, so seed variability includes subset variability.
The chosen image ids are saved in each run's `train_subset.json`.

**LR tuning** (`python -m dl_research.tune_lr`): grid search, one run per candidate (U-Net 1e-4/3e-4/1e-3; SegFormer 3e-5/6e-5/1e-4/3e-4), 10 epochs, 100% labels, seed 0, selected by best-checkpoint validation mIoU — validation only.
The winner is written to `configs/tuned_lr.json` (with the whole grid) and used for all fractions. One LR for the whole network (no separate encoder/decoder LR); the grids are small and the single-seed tuning is noisy.

**Efficiency**: parameters; GFLOPs for one 3×512×512 image (`torch.utils.flop_counter`, 1 MAC = 2 FLOPs); inference speed = mean of 50 forward passes after 10 warm-up passes, fp32, eval mode, batch 1, 512×512, with device synchronisation; training time = summed epoch wall time, **excluding validation, including data loading**. All of it is recorded together with device name and software versions, so numbers are comparable only on the same machine.

## Running
```bash
bash scripts/smoke.sh                    # synthetic data, CPU, ~1 min: checks the whole pipeline
bash scripts/run_experiments.sh          # tune LR -> train 2 models x 3 fractions x 3 seeds -> aggregate
```
Individual steps:
```bash
python -m dl_research.tune_lr --config configs/base.yaml
python -m dl_research.train --config configs/base.yaml configs/segformer_b2.yaml --set fraction=0.25 seed=1
python -m dl_research.evaluate --run experiments/segformer_b2_f0.25_s1 --split val --max-save 100
python -m dl_research.aggregate --exp-dir experiments --out results
```
Runs are resumable (checkpoint at every validation epoch; rerun the same command) and finished runs are skipped.
`--force` restarts a run; a changed config for an existing run directory is refused.

### Outputs
```
experiments/<model>_f<fraction>_s<seed>/
  config.yaml  train_subset.json  log.jsonl  metrics.json  eval_val_best.json
  checkpoints/{best,last}.pt   predictions/val_best/*.png
experiments/tune/lr<value>/...      # LR-tuning runs
results/summary.md  summary.csv  all_runs.csv  gap_vs_fraction.csv  figures/*.png
```
`metrics.json`: config, environment, parameter count, efficiency, training stats and validation mIoU, per-class IoU and F1, pixel accuracy (macro F1 `mf1`) — overall and for urban/rural. Prediction PNGs use raw LoveDA values (1–7).
`aggregate` reports mean ± sample std over seeds (n=1 is shown as such) and the SegFormer−U-Net mIoU gap paired by seed.

## Known limitations
- **Test labels are not public.** All reported numbers are on the **validation** split, which is *also* used for LR selection and best-checkpoint selection, so they are optimistically biased and not comparable to the official test leaderboard. `final_epoch_val_miou` is stored to show the effect of checkpoint selection. For test numbers, run `evaluate --split test` (writes predictions only) and submit them to the benchmark server; check its expected label encoding first.
- Few seeds (default 3) give only a rough view of variability; no significance tests are included.
- LR tuned once at 100% labels / seed 0 and reused at lower fractions.
- Rotations are multiples of 90° only; no color/scale augmentation. Single-scale, no test-time augmentation.
- Different hardware or library versions change timings; GPU determinism is best-effort (some CUDA ops are non-deterministic).
- SegFormer's decode head is randomly initialised (only the encoder is ImageNet-pretrained); U-Net's decoder likewise.
- Memory: SegFormer-B2 at batch 8 / 512² and full-image validation need a reasonably large GPU; lower `batch_size` / `eval_batch_size` if needed (this changes the protocol — change it for both models).

## Adding a dataset (e.g. DeepGlobe)
Add a module that builds a `DatasetSpec` (class names, domains, label LUTs, `scan`) and `register()`s it, like
`src/dl_research/loveda.py`, then set `dataset: <name>` in a config. Training, evaluation and aggregation use only the spec; DeepGlobe is intentionally not part of the primary comparison.

## Google Colab (T4) notebook
`segformer_vs_unet_colab.ipynb` is standalone: GPU check, `pip install`, Drive mount (outputs persist on Drive), LoveDA download,
synthetic smoke test, a short real-data pipeline check that times the GPU and prints a projected total, LR tuning, the study,
and results. `PROFILE = "quick"` (1 seed, 10 base epochs) or `"standard"` (3 seeds, 30 base epochs). Colab-specific settings:
fp16 autocast for training **and validation** (`eval_amp`; same for both models, the efficiency benchmark stays fp32) and
`prune_last_checkpoint` (only `best.pt` is kept for finished runs, so Drive does not fill up). After a disconnect, *Run all*
again: finished runs are skipped and interrupted runs resume from the last validation-epoch checkpoint. Not verified on a real
T4 session in development: the Drive mount, `pip install`, GPU fp16 paths and the Zenodo download are exercised only by Colab itself.
Both notebooks are generated by `python scripts/build_notebook.py`.

## Layout
`src/dl_research/`: `config`, `data` (registry, subsets, dataset), `loveda`, `metrics`, `models`, `efficiency`,
`train`, `evaluate`, `tune_lr`, `aggregate`, `prepare_dataset`, `utils`. `configs/`, `scripts/`, `tests/`.
