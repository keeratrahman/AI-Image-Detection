#!/usr/bin/env python3
"""Export plot-ready CSVs from runs/*/metrics.json.

    python scripts/export_plot_data.py            # writes figures/data/*.csv

Outputs (one row per line, comma-separated, missing values written as "nan"):
  curves/<run>.csv          epoch, train_loss, val_loss, val_auc, val_acc   (every run)
  curves_main.csv           epoch, <model>_val_auc, <model>_val_loss ...     (best-LR jpeg-crop runs)
  jpeg_curve.csv            jpeg_quality, <model> held-out AUC, <model> fake acc  (100 = clean)
  per_generator_auc.csv     generator, <model> ...   (clean test AUC, main runs)
  robustness_long.csv       run, model, prep, perturbation, mean/seen/near/far/held-out AUC, accs
  h4_ablation.csv           model, prep, held-out AUC, clean/q96 AUC and fake acc, drops

All numbers come straight from metrics.json, so re-running this after adding seeds or evals
regenerates every figure's data.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "figures" / "data"

# Best-LR jpeg-crop run per model (seed 0). ResNet-18 is the baseline.
MAIN = {
    "resnet18": "A_resnet18_in1k_jpeg-crop_lr3e-4_s0",
    "resnet50": "A_resnet50_in1k_jpeg-crop_lr3e-4_s0",
    "deit-s": "A_deit-s_in1k_jpeg-crop_lr3e-5_s0",
}
H4_LR = {"resnet50": "3e-4", "deit-s": "3e-5"}
PREPS = ("raw-resize", "raw-crop", "jpeg-resize", "jpeg-crop")
GENERATORS = ["sdv14", "sdv15", "wukong", "adm", "glide", "vqdm", "midjourney", "biggan"]
JPEG_QUALITIES = [100, 96, 95, 75, 50]  # 100 = clean (no compression)


def load(run):
    p = RUNS / run / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def num(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else x


def mean_fake_acc(section):
    vals = [v for v in section["per_generator_fake_acc"].values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def gen_value(section, key, gen):
    d = section.get(f"per_generator_{key}", {})
    return d.get(gen, d.get(f"test_{gen}"))  # manifests are named test_<gen>


GROUPS = {"seen": ["sdv14"], "near": ["sdv15", "wukong"],
          "far": ["adm", "glide", "vqdm", "midjourney", "biggan"]}
GROUPS["held_out"] = GROUPS["near"] + GROUPS["far"]


def group_auc(section, group):
    """Recompute from per-generator AUCs: some robustness evals were run without --seen/--near/--far."""
    vals = [gen_value(section, "auc", g) for g in GROUPS[group]]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if len(vals) == len(GROUPS[group]) else None


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[num(v) for v in r] for r in rows])
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    # 1. per-run training curves
    for d in sorted(RUNS.iterdir()):
        m = load(d.name)
        if not m or not m.get("history"):
            continue
        write(OUT / "curves" / f"{d.name}.csv",
              ["epoch", "train_loss", "val_loss", "val_auc", "val_acc"],
              [[h["epoch"], h.get("train_loss"), h.get("val_loss"), h.get("val_auc"), h.get("val_acc")]
               for h in m["history"]])

    # 2. main runs' curves side by side (wide, for one figure)
    main = {k: load(v) for k, v in MAIN.items()}
    n_epochs = max(len(m["history"]) for m in main.values() if m)
    header, rows = ["epoch"], []
    for k in main:
        header += [f"{k}_train_loss", f"{k}_val_loss", f"{k}_val_auc"]
    for e in range(n_epochs):
        row = [e + 1]
        for m in main.values():
            h = m["history"][e] if m and e < len(m["history"]) else {}
            row += [h.get("train_loss"), h.get("val_loss"), h.get("val_auc")]
        rows.append(row)
    write(OUT / "curves_main.csv", header, rows)

    # 3. held-out AUC and fake accuracy vs JPEG quality
    header = ["jpeg_quality"] + [f"{k}_held_out_auc" for k in main] + [f"{k}_fake_acc" for k in main]
    rows = []
    for q in JPEG_QUALITIES:
        aucs, accs = [], []
        for m in main.values():
            s = m["test"] if q == 100 else m.get("robustness", {}).get(f"jpeg:{q}") if m else None
            aucs.append(s.get("mean_auc_held_out") if s else None)
            accs.append(mean_fake_acc(s) if s else None)
        rows.append([q] + aucs + accs)
    write(OUT / "jpeg_curve.csv", header, rows)

    # 4. clean per-generator AUC
    write(OUT / "per_generator_auc.csv", ["generator"] + list(main),
          [[g] + [gen_value(m["test"], "auc", g) if m else None for m in main.values()] for g in GENERATORS])

    # 5. every (run, perturbation) summary, long format
    rows = []
    for d in sorted(RUNS.iterdir()):
        m = load(d.name)
        if not m or "mean_auc" not in m.get("test", {}):
            continue
        sections = [("clean", m["test"])] + sorted(m.get("robustness", {}).items())
        for name, s in sections:
            rows.append([d.name, m["model"], m["prep"], name, s.get("mean_auc"), s.get("mean_auc_seen"),
                         s.get("mean_auc_near"), s.get("mean_auc_far"), s.get("mean_auc_held_out"),
                         s.get("mean_acc"), mean_fake_acc(s)])
    write(OUT / "robustness_long.csv",
          ["run", "model", "prep", "perturbation", "mean_auc", "auc_seen", "auc_near", "auc_far",
           "auc_held_out", "mean_acc", "fake_acc"], rows)

    # 6. H4 2x2 ablation
    rows = []
    for model, lr in H4_LR.items():
        for prep in PREPS:
            m = load(f"A_{model}_in1k_{prep}_lr{lr}_s0")
            if not m:
                continue
            t, r = m["test"], m["robustness"]["jpeg:96"]
            fc, fr = mean_fake_acc(t), mean_fake_acc(r)
            rows.append([model, prep, t["mean_auc_held_out"], t["mean_auc"], r["mean_auc"],
                         t["mean_auc"] - r["mean_auc"], fc, fr, fc - fr])
    write(OUT / "h4_ablation.csv",
          ["model", "prep", "held_out_auc", "auc_clean", "auc_q96", "auc_drop",
           "fake_acc_clean", "fake_acc_q96", "fake_acc_drop"], rows)


if __name__ == "__main__":
    main()
