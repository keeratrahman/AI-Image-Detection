#!/usr/bin/env python3
"""H1 confidence interval: paired bootstrap of the DeiT-S minus ResNet-50 AUC gap on the far generators.

    python scripts/bootstrap_ci.py \\
        --vit runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s{0,1,2} \\
        --cnn runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s{0,1,2}

Needs per-image predictions from `eval.py --save-preds` in <run-dir>/preds/<condition>/<generator>.npz.

  * Point estimate: per-seed AUC averaged over seeds, then the mean over the group's generators.
    Gap = ViT - CNN, in AUC points (x100).
  * Bootstrap: resample images with replacement within each generator, separately for real and
    fake, using the SAME indices for both models and every seed (paired). CI = 2.5th-97.5th
    percentile of the resampled gaps.
  * Seed std: the gap computed per seed pair (ViT seed k vs CNN seed k), std across pairs.
  * H1 rule (midterm): supported if gap >= 3 AUC points and the 95% CI excludes 0.
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.metrics import roc_auc  # noqa: E402

FAR = ["test_adm", "test_glide", "test_vqdm", "test_midjourney", "test_biggan"]


def load_preds(run_dirs, generators, condition):
    """{generator: (logits [n_runs, n_images], labels [n_images], paths [n_images])}."""
    out = {}
    for g in generators:
        files = [Path(r) / "preds" / condition / f"{g}.npz" for r in run_dirs]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit("Missing predictions (run eval.py --save-preds first):\n  " + "\n  ".join(missing))
        data = [np.load(f) for f in files]
        paths, labels = data[0]["paths"], data[0]["labels"]
        for f, d in zip(files[1:], data[1:]):
            if len(d["paths"]) != len(paths) or not (d["paths"] == paths).all():
                raise SystemExit(f"Image order in {f} differs from {files[0]}; the paired bootstrap needs the same order.")
        out[g] = (np.stack([d["logits"] for d in data]), labels, paths)
    return out


def per_seed_gen_auc(preds, idx=None):
    """AUC array [n_runs, n_generators], optionally on resampled indices idx[g]."""
    cols = []
    for g, (logits, labels, _) in preds.items():
        i = slice(None) if idx is None else idx[g]
        cols.append([roc_auc(labels[i], run_logits[i]) for run_logits in logits])
    return np.array(cols).T


def group_auc(preds, idx=None):
    """Per-seed AUC averaged over seeds, then the mean over generators."""
    return per_seed_gen_auc(preds, idx).mean(axis=0).mean()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vit", nargs="+", required=True, help="DeiT-S run dirs (one per seed)")
    p.add_argument("--cnn", nargs="+", required=True, help="ResNet-50 run dirs (one per seed, same seed order)")
    p.add_argument("--generators", nargs="+", default=FAR, help="generator group (default: the 5 far generators)")
    p.add_argument("--group-name", default="far")
    p.add_argument("--condition", default="clean", help="preds/<condition>/ folder: clean, aligned, jpeg_75, ...")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    p.add_argument("--threshold", type=float, default=3.0, help="H1 minimum gap in AUC points")
    p.add_argument("--out", default="results/h1_ci.md", help="markdown summary to write ('' to skip)")
    return p.parse_args()


def main():
    args = parse_args()
    vit = load_preds(args.vit, args.generators, args.condition)
    cnn = load_preds(args.cnn, args.generators, args.condition)
    for g in args.generators:
        if not (vit[g][2] == cnn[g][2]).all():
            raise SystemExit(f"{g}: ViT and CNN predictions are on different images or in a different order.")

    # point estimate
    vit_auc = per_seed_gen_auc(vit)           # [n_vit_seeds, n_gen]
    cnn_auc = per_seed_gen_auc(cnn)
    gap = 100 * (vit_auc.mean(0).mean() - cnn_auc.mean(0).mean())

    # seed-to-seed std of the gap (pairs ViT seed k with CNN seed k)
    if len(args.vit) == len(args.cnn) and len(args.vit) > 1:
        seed_gaps = 100 * (vit_auc.mean(1) - cnn_auc.mean(1))
        seed_std = float(np.std(seed_gaps, ddof=1))
    else:
        seed_gaps, seed_std = None, float("nan")

    # paired bootstrap: one set of indices per resample, shared by both models and all seeds
    rng = np.random.default_rng(args.seed)
    strata = {g: (np.flatnonzero(vit[g][1] < 0.5), np.flatnonzero(vit[g][1] >= 0.5)) for g in args.generators}
    gaps = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = {g: np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
               for g, (real, fake) in strata.items()}
        gaps[b] = 100 * (group_auc(vit, idx) - group_auc(cnn, idx))
        if (b + 1) % 1000 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", file=sys.stderr, flush=True)
    lo, hi = np.percentile(gaps, [2.5, 97.5])

    supported = gap >= args.threshold and lo > 0
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    headline = f"gap = {gap:.1f} [{lo:.1f}, {hi:.1f}]  seed std = {seed_std:.1f}"

    lines = [
        f"# H1: DeiT-S minus ResNet-50 AUC gap ({args.group_name} generators, {args.condition} test)",
        "",
        f"**{headline}** (AUC points; 95% paired bootstrap CI, {args.n_boot} resamples)",
        "",
        f"H1 rule: gap >= {args.threshold:g} points and CI excludes 0 -> **{verdict}**",
        "",
        "| generator | DeiT-S AUC | ResNet-50 AUC | gap (pts) |",
        "|---|---|---|---|",
    ]
    for j, g in enumerate(args.generators):
        v, c = vit_auc[:, j].mean(), cnn_auc[:, j].mean()
        lines.append(f"| {g} | {v:.4f} | {c:.4f} | {100 * (v - c):+.1f} |")
    lines.append(f"| **mean** | {vit_auc.mean(0).mean():.4f} | {cnn_auc.mean(0).mean():.4f} | {gap:+.1f} |")
    lines += ["", f"Seeds: {len(args.vit)} DeiT-S, {len(args.cnn)} ResNet-50."]
    if seed_gaps is not None:
        lines.append("Per-seed gaps: " + ", ".join(f"{x:+.1f}" for x in seed_gaps))
    lines += ["", "Runs:"] + [f"- ViT: `{r}`" for r in args.vit] + [f"- CNN: `{r}`" for r in args.cnn]
    lines += ["", f"Command: `python {' '.join(sys.argv)}`",
              f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
    text = "\n".join(lines) + "\n"

    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
