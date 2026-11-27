#!/usr/bin/env python3
"""Task 6, step 2: logistic-regression probe on frozen CLIP features, evaluated like every other run.

    python scripts/clip_probe.py                       # reads features/clip-l14/, writes runs_final/A_clip-l14-probe_jpeg-crop_s0/

  * Fits on A_train only; picks C from {0.01, 0.1, 1, 10} by A_val AUC (ties within 0.0005 broken
    by lowest A_val log loss, the same rule as the LR sweep). Features are L2-normalized first.
  * The probe's decision function is a log-odds logit, so "logit > 0 = fake" matches eval.py.
  * Scores each test generator with eval.score / eval.summarize and writes metrics.json in the
    same format as the trained runs (test, plus aligned_test / robustness for any extra feature
    conditions found), so compare.py, aggregate_seeds.py and bootstrap_ci.py work unchanged.
  * Saves per-image predictions to <run-dir>/preds/<condition>/<generator>.npz (as eval.py --save-preds).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.metrics import roc_auc  # noqa: E402
from aidet.naming import TEST_GENERATORS  # noqa: E402
from eval import clean, git_commit, save_preds, score, summarize  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0]
TIE = 0.0005
GROUPS = {
    "seen": ["test_sdv14"],
    "near": ["test_sdv15", "test_wukong"],
    "far": ["test_adm", "test_glide", "test_vqdm", "test_midjourney", "test_biggan"],
}
GROUPS["held_out"] = GROUPS["near"] + GROUPS["far"]


def load(path):
    d = np.load(path)
    x = d["feats"].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x, d["labels"], d["paths"], str(d["manifest"])


def log_loss(labels, logits):
    # mean binary cross-entropy from logits, numerically stable
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def spec_from_condition(cond):
    """Folder name -> metrics key: jpeg_75 -> jpeg:75, rescale_0.5 -> rescale:0.5."""
    kind, _, arg = cond.partition("_")
    return f"{kind}:{arg}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="features/clip-l14", help="output folder of clip_features.py")
    p.add_argument("--run-dir", default="runs_final/A_clip-l14-probe_jpeg-crop_s0")
    p.add_argument("--max-iter", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    from sklearn.linear_model import LogisticRegression

    feat_dir = Path(args.features)
    xtr, ytr, _, _ = load(feat_dir / "clean" / "A_train.npz")
    xva, yva, _, _ = load(feat_dir / "clean" / "A_val.npz")
    print(f"A_train {xtr.shape}, A_val {xva.shape}")

    grid = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=args.max_iter).fit(xtr, ytr)
        lv = clf.decision_function(xva)
        grid[c] = {"val_auc": roc_auc(yva, lv), "val_loss": log_loss(yva, lv), "clf": clf}
        print(f"  C={c:<5g} val_auc={grid[c]['val_auc']:.6f} val_loss={grid[c]['val_loss']:.4f}")
    top = max(g["val_auc"] for g in grid.values())
    best_c = min((c for c in C_GRID if grid[c]["val_auc"] >= top - TIE), key=lambda c: grid[c]["val_loss"])
    clf = grid[best_c]["clf"]
    print(f"Best C = {best_c:g} (highest val AUC; ties within {TIE} -> lowest val loss)")

    run_dir = Path(args.run_dir)
    preds_dir = run_dir / "preds"
    conditions = sorted(d.name for d in feat_dir.iterdir() if d.is_dir())
    blocks, manifests = {}, {}
    for cond in conditions:
        results = {}
        for g in TEST_GENERATORS:
            f = feat_dir / cond / f"{g}.npz"
            if not f.exists():
                continue
            x, y, paths, manifest = load(f)
            logits = clf.decision_function(x)
            results[g] = score(logits, y)
            save_preds(preds_dir, cond, g, logits.astype(np.float32), y, manifest, list(paths))
            if cond == "clean":
                manifests[g] = manifest
        if len(results) != len(TEST_GENERATORS):
            print(f"  skipping condition '{cond}': has {len(results)}/{len(TEST_GENERATORS)} test generators")
            continue
        blocks[cond] = summarize(results, GROUPS)
        blocks[cond]["jpeg_aligned_at_test"] = cond == "aligned"
        print(f"  [{cond}] mean auc={blocks[cond]['mean_auc']:.4f}  far={blocks[cond]['mean_auc_far']:.4f}  "
              f"mean acc={blocks[cond]['mean_acc']:.4f}")
    if "clean" not in blocks:
        raise SystemExit(f"No complete clean test features in {feat_dir / 'clean'}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metrics = {
        "schema_version": 1,
        "run_name": run_dir.name,
        "protocol": "A",
        "model": "clip-l14-probe",
        "init": "openai",
        "prep": "jpeg-crop",
        "lr": None,  # no gradient training; the probe's hyperparameter is C (see probe)
        "seed": 0,
        "device": "cpu",
        "val": {"auc": grid[best_c]["val_auc"], "loss": grid[best_c]["val_loss"]},
        "probe": {
            "features": "open_clip ViT-L-14 (openai), frozen image embedding, L2-normalized",
            "classifier": "sklearn LogisticRegression (lbfgs, L2)",
            "C": best_c,
            "selection": f"highest A_val AUC; ties within {TIE} broken by lowest A_val log loss",
            "grid": {f"{c:g}": {"val_auc": g["val_auc"], "val_loss": g["val_loss"]} for c, g in grid.items()},
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "feature_dim": int(xtr.shape[1]),
        },
        "test": blocks["clean"],
        "eval": {"timestamp": now, "git_commit": git_commit(), "manifests": manifests, "device": "cpu"},
        "git_commit": git_commit(),
        "timestamp": now,
    }
    if "aligned" in blocks:
        metrics["aligned_test"] = blocks["aligned"]
    rob = {spec_from_condition(c): b for c, b in blocks.items() if c not in ("clean", "aligned")}
    if rob:
        metrics["robustness"] = rob

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(clean(metrics), f, indent=2)
    print(f"Wrote {run_dir / 'metrics.json'} and preds under {preds_dir}")


if __name__ == "__main__":
    main()
