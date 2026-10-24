# AI-Image-Detection

Binary classifier that detects AI-generated vs. real images.

## Layout

| Path | What it does |
|---|---|
| `aidet/` | Core package |
| `aidet/data.py` | Dataset and DataLoader code (reads images, applies transforms, builds splits) |
| `aidet/models.py` | Model definitions (e.g. ResNet18, from scratch or pretrained) |
| `aidet/metrics.py` | Evaluation metrics (accuracy, precision/recall, F1, etc.) |
| `aidet/naming.py` | Builds run names from config values, e.g. `A_resnet18_scratch_jpeg-crop_lr1e-3_s0` |
| `train.py` | Training entry point: `python train.py --config <config.yaml>` |
| `make_configs.py` | Generates the experiment YAML configs |
| `configs/` | Experiment configs (`configs/smoke/` = quick pipeline tests) |
| `runs_smoke/` | Output of smoke runs (config, metrics, checkpoint). Local only, not tracked in git |
| `requirements.txt` | Python dependencies: `pip install -r requirements.txt` |

## Quick start

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/smoke/<config>.yaml
```

## Not yet implemented

- **Real dataset (in progress, Nhi):** will be plugged in through `aidet/data.py`, with the dataset path and split set in the YAML configs. Expected folder layout is TBD (to be agreed with Nhi and documented here).
- **Full experiment sweep:** after the data lands, expand `make_configs.py` to generate the full config grid into `configs/`.
- **Standalone evaluation script:** `eval.py` (planned) will load a checkpoint and report test-set metrics using `aidet/metrics.py`.
