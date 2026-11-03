# AI-Image-Detection

Binary classifier that detects AI-generated vs. real images. We compare ResNet-50, ResNet-18 and DeiT-S
trained on GenImage SD v1.4 and tested on 8 generators. See `docs/PLAN.md` for hypotheses and setup.

## Layout

| Path | What it does |
|---|---|
| `aidet/` | Core package |
| `aidet/data.py` | DataLoader construction; picks the stub or GenImage dataset from the config |
| `aidet/genimage.py` | GenImage dataset: reads a JSONL manifest, JPEG alignment, crop/resize to 224 |
| `aidet/models.py` | Model registry: `resnet18`, `resnet50`, `deit-s` (ImageNet-1k or scratch init), one output logit |
| `aidet/metrics.py` | ROC-AUC. Accuracy, precision, recall and F1 are computed in `eval.py` |
| `aidet/naming.py` | Builds run names from config values, e.g. `A_resnet50_in1k_jpeg-crop_lr3e-4_s0` |
| `train.py` | Training entry point: `python train.py --config <config.yaml>`. Auto-selects CUDA, then Apple GPU (MPS), then CPU |
| `eval.py` | Evaluates a trained run on per-generator test manifests and writes results into its `metrics.json` |
| `make_configs.py` | Generates the experiment YAML configs |
| `compare.py` | Prints per-generator AUC, group means and robustness for several runs side by side |
| `ablation.py` | Prints the H4 ablation table (JPEG alignment x crop/resize) |
| `bench.py` | Times data loading and the model step, to estimate epoch time |
| `scripts/setup_local.sh` | One-time local setup: venv, pretrained weights, LR-sweep configs |
| `scripts/run_lr_sweep.sh` | Runs the configs in `configs/lr_sweep/`, skipping finished runs |
| `scripts/best_lr.py` | Picks each model's best LR by seen-generator validation AUC |
| `scripts/pace/` | Earlier PACE-ICE (SLURM) setup, no longer used |
| `scripts/sample_zip.py` | Extracts a seeded random sample from a downloaded GenImage zip (Kaggle mirror) |
| `scripts/make_subset.py` | Samples a balanced GenImage subset and writes protocol A/B + per-generator test manifests |
| `docs/PLAN.md` | Hypotheses, fixed setup, mid plan and final plan |
| `docs/DATA.md` | Step-by-step: download GenImage and build the subset |
| `configs/` | Experiment configs (`configs/lr_sweep/` = LR sweep and ablation, `configs/smoke/` = quick pipeline tests) |
| `runs/` | One folder per run. `metrics.json` and `config.yaml` are tracked; checkpoints (`best.pt`) and W&B files are not |
| `results/` | Written-up results: `protocol_a_seed0.md` (sweeps and evals), `h4_ablation_seed0.md` (ablation) |
| `requirements.txt` | Python dependencies: `pip install -r requirements.txt` |

Data and checkpoints live on shared storage, never in git. Configs, manifests and `metrics.json` files are in git.

## Quick start

On a Mac (Apple GPU), from the repo root. The script expects the data in `../genimage_subset`; set `DATA_ROOT` for another location:

```
bash scripts/setup_local.sh        # venv, pretrained weights, LR-sweep configs
wandb login                        # or run with WANDB_MODE=disabled
caffeinate -i bash scripts/run_lr_sweep.sh "*resnet50*"    # pattern is optional
python scripts/best_lr.py runs     # best LR per model
```

Alternative setup (any machine):

```
uv venv ~/.venvs/aidet --python 3.11 && source ~/.venvs/aidet/bin/activate   # keep the venv off exFAT drives
uv pip install -r requirements.txt
python train.py --config configs/smoke/A_resnet18_scratch_jpeg-crop_lr1e-3_s0.yaml
```

For real data, follow [docs/DATA.md](docs/DATA.md).

## Evaluating a run

```
python eval.py --run-dir runs/<run> --manifests ../genimage_subset/manifests/test_*.jsonl \
  --seen test_sdv14 --near test_sdv15 test_wukong \
  --far test_adm test_glide test_vqdm test_midjourney test_biggan \
  --robustness jpeg:95 jpeg:75 jpeg:50 blur:1 blur:2 rescale:0.5
python compare.py runs/<resnet50_run> runs/<deit-s_run> runs/<resnet18_run>
```

`eval.py` writes per-generator AUC and accuracy, group means (seen, near, far, held-out) and robustness into
the run's `metrics.json`. Generator names are the manifest file names.

## Status

Done (Protocol A, seed 0): LR sweeps for ResNet-50, ResNet-18 and DeiT-S, evals on all 8 generators with robustness, and the
H4 ablation for ResNet-50 and DeiT-S. See `results/`.

## Next (see `docs/PLAN.md`)

- Seeds 1 and 2 for the main runs, with confidence intervals for H1.
- H3 scratch-init runs (`make_configs.py --init scratch`).
- Protocol B (H5) and the CLIP linear probe.
- `cnn-small` in `aidet/models.py`, if not yet registered.
- A debiased test set to separate the JPEG shortcut from real artifacts in the ablation.
