#!/usr/bin/env python3
"""Generate one YAML config per run, named {protocol}_{model}_{init}_{prep}_lr{lr}_s{seed}.

Examples:
    # LR sweep for ResNet-50 on Protocol A (raw-resize sanity run + jpeg-crop main run)
    python make_configs.py --models resnet50 --manifest-dir /shared/genimage/manifests

    # Once the best LR is known, add the full H4 2x2 at that LR
    python make_configs.py --models resnet50 --h4-lr 1e-4 --manifest-dir /shared/genimage/manifests

    # Smoke configs on stub data
    python make_configs.py --data stub --init scratch --models resnet18 --lrs 1e-3 \\
        --preps jpeg-crop --epochs 3 --batch-size 32 --num-workers 0 --wandb-mode disabled \\
        --output-dir runs_smoke --out configs/smoke
"""
import argparse
from pathlib import Path

import yaml

from aidet.naming import INITS, PREPS, run_name_from_cfg

# Same 3-LR budget for every model (team fairness rule). Placeholder values: agree as a team.
DEFAULT_LRS = [3e-4, 1e-4, 3e-5]
SWEEP_PREPS = ["raw-resize", "jpeg-crop"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="configs")
    p.add_argument("--protocol", default="A")
    p.add_argument("--data", choices=["genimage", "stub"], default="genimage")
    p.add_argument("--manifest-dir", default="/path/to/shared/manifests")
    p.add_argument("--models", nargs="+", default=["resnet50"])
    p.add_argument("--init", choices=INITS, default="in1k")
    p.add_argument("--preps", nargs="+", choices=PREPS, default=SWEEP_PREPS)
    p.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--h4-lr", type=float, help="add all four H4 prep cells at this LR")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-dir", default="/path/to/shared/runs", help="shared storage, never git")
    p.add_argument("--project", default="ai-image-detection")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def data_block(args):
    if args.data == "stub":
        return {"name": "stub", "stub_train_size": 128, "stub_val_size": 64}
    m = args.manifest_dir  # placeholder file names: align with Nhi's manifests
    return {
        "name": "genimage",
        "train_manifest": f"{m}/{args.protocol}_train.jsonl",
        "val_manifest": f"{m}/{args.protocol}_val.jsonl",
    }


def make_cfg(args, model, prep, lr, seed):
    return {
        "protocol": args.protocol, "model": model, "init": args.init, "prep": prep,
        "lr": lr, "seed": seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "weight_decay": args.weight_decay, "warmup_frac": 0.05, "amp": True,
        "num_workers": args.num_workers,
        "data": data_block(args),
        "output_dir": args.output_dir,
        "wandb": {"project": args.project, "mode": args.wandb_mode},
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for model in args.models:
        for seed in args.seeds:
            jobs += [(model, prep, lr, seed) for prep in args.preps for lr in args.lrs]
            if args.h4_lr is not None:
                jobs += [(model, prep, args.h4_lr, seed) for prep in PREPS]

    written = {}
    for model, prep, lr, seed in jobs:
        cfg = make_cfg(args, model, prep, lr, seed)
        name = run_name_from_cfg(cfg)
        if name in written:
            continue
        with open(out / f"{name}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written[name] = True

    print(f"Wrote {len(written)} configs to {out}/")
    for name in written:
        print(f"  {name}.yaml")


if __name__ == "__main__":
    main()
