# AI-Image-Detection

Binary classifier that detects AI-generated vs. real images.

## Layout

| Path | What it does |
|---|---|
| `aidet/` | Core package |
| `aidet/data.py` | DataLoader construction; picks the stub or GenImage dataset from the config |
| `aidet/genimage.py` | GenImage dataset: reads a JSONL manifest, JPEG alignment, crop/resize to 224 |
| `aidet/models.py` | Model definitions (e.g. ResNet18, from scratch or pretrained) |
| `aidet/metrics.py` | Evaluation metrics (accuracy, precision/recall, F1, etc.) |
| `aidet/naming.py` | Builds run names from config values, e.g. `A_resnet18_scratch_jpeg-crop_lr1e-3_s0` |
| `train.py` | Training entry point: `python train.py --config <config.yaml>` |
| `make_configs.py` | Generates the experiment YAML configs |
| `scripts/sample_zip.py` | Extracts a seeded random sample from a downloaded GenImage zip (Kaggle mirror) |
| `scripts/make_subset.py` | Samples a balanced GenImage subset and writes protocol A/B + per-generator test manifests |
| `docs/PLAN.md` | Hypotheses, fixed setup, mid plan and final plan |
| `docs/DATA.md` | Step-by-step: download GenImage and build the subset |
| `configs/` | Experiment configs (`configs/smoke/` = quick pipeline tests) |
| `runs_smoke/` | Output of smoke runs (config, metrics, checkpoint). Local only, not tracked in git |
| `requirements.txt` | Python dependencies: `pip install -r requirements.txt` |

## Quick start

```
uv venv ~/.venvs/aidet --python 3.11 && source ~/.venvs/aidet/bin/activate   # keep the venv off exFAT drives
uv pip install -r requirements.txt
python train.py --config configs/smoke/A_resnet18_scratch_jpeg-crop_lr1e-3_s0.yaml
```

For real data, follow [docs/DATA.md](docs/DATA.md).

## Not yet implemented

- **Models:** `deit-s` and `cnn-small` still need to be registered in `aidet/models.py` (see `docs/PLAN.md`).
- **Full experiment sweep:** after the data lands, expand `make_configs.py` to generate the full config grid into `configs/`.
- **Standalone evaluation script:** `eval.py` (planned) will load a checkpoint and report per-generator test metrics on `test_<generator>.jsonl`, plus robustness perturbations (`GenImageDataset(extra_transform=...)`).
