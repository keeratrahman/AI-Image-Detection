#!/usr/bin/env python3
"""Train one run from one YAML config and write metrics.json.

    python train.py --config configs/A_resnet50_in1k_raw-resize_lr1e-4_s0.yaml

Model selection uses validation AUC on seen generators only. Held-out/test evaluation
(per-generator AUC, robustness) belongs to eval.py, which loads best.pt.
"""
import argparse
import datetime
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from aidet.data import build_dataloaders
from aidet.metrics import roc_auc
from aidet.models import build_model
from aidet.naming import run_name_from_cfg

# PROVISIONAL: align with Ahmed A's eval.py before CP0.
SCHEMA_VERSION = "0.1-provisional"

REQUIRED = ("protocol", "model", "init", "prep", "lr", "seed", "epochs", "batch_size", "data")
DEFAULTS = {
    "weight_decay": 1e-2,
    "warmup_frac": 0.05,
    "amp": True,
    "num_workers": 4,
    "output_dir": "runs",
    "wandb": {"project": "ai-image-detection", "mode": "online"},
}


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    for key, default in DEFAULTS.items():
        if isinstance(default, dict):
            cfg[key] = {**default, **cfg.get(key, {})}
        else:
            cfg.setdefault(key, default)
    name = run_name_from_cfg(cfg)
    if path.stem != name:
        raise SystemExit(f"Config file is named '{path.stem}.yaml' but its contents imply '{name}.yaml'")
    cfg["run_name"] = name
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def make_scheduler(optimizer, total_steps, warmup_frac):
    warm = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, amp):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        with autocast(device, amp):
            logits = model(x).reshape(-1)
        loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    total, n = 0.0, 0
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        with autocast(device, amp):
            logits = model(x).reshape(-1)
        logits = logits.float()
        total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total / max(n, 1),
        "auc": roc_auc(labels, logits),
        "acc": float(((logits > 0) == (labels > 0.5)).mean()),
    }


def _clean(obj):
    """Make NaN/inf JSON-safe (null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, help="override epochs (does not change the run name)")
    p.add_argument("--output-dir", help="override output_dir (use shared storage, never git)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.add_argument("--device")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.wandb_mode:
        cfg["wandb"]["mode"] = args.wandb_mode

    seed_everything(cfg["seed"])
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple-silicon laptops: fine for smoke tests only
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg["model"], cfg["init"]).to(device)
    criterion = nn.BCEWithLogitsLoss()  # label 1 = fake
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg["epochs"] * len(train_loader), cfg["warmup_frac"])

    import wandb

    run = wandb.init(
        project=cfg["wandb"]["project"], name=cfg["run_name"], config=cfg,
        mode=cfg["wandb"]["mode"], dir=str(run_dir),
        tags=[cfg["protocol"], cfg["model"], cfg["init"], cfg["prep"]],
    )

    best_score, best_epoch, best_val = -1.0, 0, None
    ckpt_path = run_dir / "best.pt"
    history = []
    print(f"[{cfg['run_name']}] device={device} train_batches={len(train_loader)} epochs={cfg['epochs']}")
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg["amp"])
        val = evaluate(model, val_loader, criterion, device, cfg["amp"])
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        run.log({
            "epoch": epoch, "train/loss": train_loss, "val/loss": val["loss"],
            "val/auc": val["auc"], "val/acc": val["acc"], "lr": scheduler.get_last_lr()[0],
        })
        score = -1.0 if math.isnan(val["auc"]) else val["auc"]
        if score > best_score:
            best_score, best_epoch, best_val = score, epoch, val
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val}, ckpt_path)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
              f"val_auc={val['auc']:.4f} val_acc={val['acc']:.4f}")

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_name": cfg["run_name"],
        "protocol": cfg["protocol"], "model": cfg["model"], "init": cfg["init"],
        "prep": cfg["prep"], "lr": cfg["lr"], "seed": cfg["seed"],
        "best_epoch": best_epoch,
        "val": best_val,
        "history": history,
        "test": {"per_generator_auc": {}},  # filled in by eval.py
        "checkpoint": str(ckpt_path),
        "git_commit": git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(_clean(metrics), f, indent=2)
    run.finish()
    print(f"[{cfg['run_name']}] done. best epoch {best_epoch}, val_auc={best_val['auc']:.4f}. "
          f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
