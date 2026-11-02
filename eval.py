#!/usr/bin/env python3
"""Evaluate a trained run on the test generators and write the results into its metrics.json.

    python eval.py --run-dir runs/A_resnet50_in1k_jpeg-crop_lr1e-4_s0 \\
        --manifests sd14=manifests/test_sd14.jsonl adm=manifests/test_adm.jsonl ...
    python eval.py --run-dir <run> --manifests manifests/test/          # every *.jsonl in the folder
    python eval.py --run-dir <run> --manifests ... --robustness jpeg:75 blur:2 rescale:0.5

Each manifest is ONE generator's test set (its fakes plus real images). The generator name is
NAME in NAME=PATH, otherwise the file name without its extension.

What it does
  * loads <run-dir>/best.pt (saved by train.py) and rebuilds the model from the saved config
  * tests on UNTOUCHED images (jpeg_aligned=False): the JPEG re-encoding is a training-time
    fix and must not be applied at test time
  * uses the same crop-vs-resize handling as the run's prep (center crop / resize, no flip)
  * computes AUC (main metric) and accuracy at logit 0 per generator, plus their means
  * optional robustness: the same test with a perturbation applied to every image first
  * merges everything into <run-dir>/metrics.json, keeping the fields train.py wrote

Output fields (every report number should come from these)
  test.per_generator_auc / per_generator_acc / per_generator_n / mean_auc / mean_acc
  robustness.<perturbation>.per_generator_auc / per_generator_acc / mean_auc / mean_acc
  eval.{timestamp, git_commit, checkpoint_epoch, manifests}

GenImage's published 72.1% is an accuracy, so compare mean_acc (not mean_auc) against it.
"""
import argparse
import datetime
import io
import json
import math
import os
import subprocess
from pathlib import Path

# Apple GPUs (MPS): run any op MPS doesn't support on the CPU instead of crashing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from aidet.genimage import GenImageDataset
from aidet.metrics import roc_auc
from aidet.models import build_model


# ---------------------------------------------------------------- perturbations (PIL -> PIL)
# Classes, not lambdas, so DataLoader workers can pickle them (macOS uses spawn).
class JpegCompress:
    def __init__(self, quality):
        self.quality = int(quality)

    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.sigma))


class Rescale:
    """Downscale by `factor` (0 < factor < 1), then upscale back to the original size."""

    def __init__(self, factor):
        self.factor = float(factor)

    def __call__(self, img):
        w, h = img.size
        small = img.resize((max(1, round(w * self.factor)), max(1, round(h * self.factor))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)


PERTURBATIONS = {"jpeg": JpegCompress, "blur": GaussianBlur, "rescale": Rescale}


def make_perturbation(spec):
    kind, _, arg = spec.partition(":")
    if kind not in PERTURBATIONS or not arg:
        raise SystemExit(f"Bad perturbation '{spec}'. Use jpeg:<quality>, blur:<sigma>, rescale:<factor>")
    return PERTURBATIONS[kind](arg)


# ---------------------------------------------------------------- core
def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_run(run_dir, device):
    ckpt_path = Path(run_dir) / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found. Has this run finished training?")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    # init="scratch": the real weights come from the checkpoint, so skip the pretrained download.
    model = build_model(cfg["model"], "scratch")
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt


@torch.no_grad()
def predict(model, loader, device):
    logits, labels = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True)).reshape(-1).float()  # fp32 on every device
        logits.append(out.cpu())
        labels.append(y.float())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def score(logits, labels):
    return {
        "auc": roc_auc(labels, logits),
        "acc": float(((logits > 0) == (labels > 0.5)).mean()),  # logit 0 == probability 0.5
        "n": int(len(labels)),
        "n_fake": int((labels > 0.5).sum()),
    }


def evaluate_generators(model, prep, manifests, device, batch_size, num_workers, perturbation=None):
    results = {}
    for name, path in manifests.items():
        ds = GenImageDataset(path, prep=prep, train=False, jpeg_aligned=False, extra_transform=perturbation)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        results[name] = score(*predict(model, loader, device))
        r = results[name]
        print(f"  {name:<14} auc={r['auc']:.4f} acc={r['acc']:.4f}  (n={r['n']}, fake={r['n_fake']})", flush=True)
    return results


def mean_ignoring_nan(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(results):
    return {
        "per_generator_auc": {k: v["auc"] for k, v in results.items()},
        "per_generator_acc": {k: v["acc"] for k, v in results.items()},
        "per_generator_n": {k: v["n"] for k, v in results.items()},
        "mean_auc": mean_ignoring_nan([v["auc"] for v in results.values()]),
        "mean_acc": mean_ignoring_nan([v["acc"] for v in results.values()]),
    }


# ---------------------------------------------------------------- io
def parse_manifests(items):
    manifests = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = None, item
        p = Path(path)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        if not files:
            raise SystemExit(f"No .jsonl manifests in {p}")
        for f in files:
            if not f.exists():
                raise SystemExit(f"Manifest not found: {f}")
            key = name if (name and len(files) == 1) else f.stem
            if key in manifests:
                raise SystemExit(f"Duplicate generator name '{key}'")
            manifests[key] = str(f)
    return manifests


def clean(obj):
    """NaN/inf -> null so metrics.json stays valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="run folder containing best.pt and metrics.json")
    p.add_argument("--manifests", nargs="+", required=True,
                   help="test manifests: PATH, NAME=PATH, or a folder of .jsonl files")
    p.add_argument("--robustness", nargs="*", default=[], metavar="SPEC",
                   help="perturbations to test too: jpeg:<q>, blur:<sigma>, rescale:<factor>")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", help="cuda, mps or cpu (default: auto-detect in that order)")
    p.add_argument("--out", help="metrics.json to update (default: <run-dir>/metrics.json)")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    manifests = parse_manifests(args.manifests)
    perturbations = {spec: make_perturbation(spec) for spec in args.robustness}

    model, cfg, ckpt = load_run(args.run_dir, device)
    run_name = cfg.get("run_name", Path(args.run_dir).name)
    print(f"[{run_name}] device={device} prep={cfg['prep']} generators={list(manifests)}")

    print("Clean test:")
    clean_results = evaluate_generators(model, cfg["prep"], manifests, device, args.batch_size, args.num_workers)
    test = summarize(clean_results)
    test["jpeg_aligned_at_test"] = False
    print(f"  mean auc={test['mean_auc']:.4f}  mean acc={test['mean_acc']:.4f}")

    robustness = {}
    for spec, pert in perturbations.items():
        print(f"Robustness {spec}:")
        res = evaluate_generators(model, cfg["prep"], manifests, device, args.batch_size, args.num_workers, pert)
        robustness[spec] = summarize(res)
        print(f"  mean auc={robustness[spec]['mean_auc']:.4f}  mean acc={robustness[spec]['mean_acc']:.4f}")

    out_path = Path(args.out) if args.out else Path(args.run_dir) / "metrics.json"
    metrics = json.load(open(out_path)) if out_path.exists() else {"run_name": run_name}
    metrics["test"] = test
    if robustness:  # keep earlier perturbations from other invocations
        metrics.setdefault("robustness", {}).update(robustness)
    metrics["eval"] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint_epoch": ckpt.get("epoch"),
        "manifests": manifests,
        "device": str(device),
    }
    with open(out_path, "w") as f:
        json.dump(clean(metrics), f, indent=2)
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()
