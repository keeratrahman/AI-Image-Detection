#!/usr/bin/env python3
"""Mean ± std over seeds for the final report: Table 2, per-generator AUC, robustness, Table 3, Figure 1.

    python scripts/aggregate_seeds.py                                   # runs_final/*/
    python scripts/aggregate_seeds.py 'runs/*/' --out /tmp/midterm.md --fig ''   # seed-0 M5 runs = midterm tables

Runs are grouped by (model, init, prep, lr). Each cell is the mean ± sample std over that group's
seeds (a single seed prints the bare value). Runs without test results are skipped;
missing results (e.g. no aligned test yet) print as -.
Table 2 and Figure 1 use the jpeg-crop runs; Table 3 lists every model run with more than one prep.
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.naming import PREPS, TEST_GENERATORS, fmt_lr  # noqa: E402

MODEL_NAMES = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "deit-s": "DeiT-S", "cnn-small": "CNN-small"}
ROBUSTNESS = ["jpeg:95", "jpeg:75", "jpeg:50", "jpeg:96", "blur:1", "blur:2", "rescale:0.5"]
MAIN_PREP = "jpeg-crop"


def load_groups(patterns):
    groups = defaultdict(list)
    for pattern in patterns:
        for d in sorted(glob.glob(pattern)):
            f = Path(d) / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            if m.get("test", {}).get("per_generator_auc"):
                groups[(m["model"], m["init"], m["prep"], m["lr"])].append(m)
    if not groups:
        raise SystemExit(f"No evaluated runs (metrics.json with a test block) match {patterns}")
    for runs in groups.values():
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"Duplicate seeds {seeds} for {runs[0]['run_name']}; pass each run dir once")
        runs.sort(key=lambda r: r["seed"])
    return groups


def values(runs, fn, scale=1.0):
    out = []
    for r in runs:
        try:
            v = fn(r)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(scale * v)
    return np.array(out, dtype=float)


def mean_fake_acc(block):
    return float(np.mean(list(block["per_generator_fake_acc"].values())))


def fmt(v, digits=3):
    if v.size == 0:
        return "-"
    if v.size == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def model_order(key):
    model, init, prep, lr = key
    rank = list(MODEL_NAMES).index(model) if model in MODEL_NAMES else len(MODEL_NAMES)
    return rank, model, init, PREPS.index(prep) if prep in PREPS else len(PREPS), -lr


def label(key, keys):
    model, init, _, lr = key
    name = MODEL_NAMES.get(model, model)
    if init != "in1k":
        name += f" ({init})"
    if len({k[3] for k in keys if k[:2] == (model, init)}) > 1:
        name += f" lr{fmt_lr(lr)}"
    return name


def main_tables(groups):
    keys = sorted((k for k in groups if k[2] == MAIN_PREP), key=model_order)
    if not keys:
        return [], keys
    t2 = []
    for k in keys:
        runs = groups[k]
        test = lambda field: values(runs, lambda r: r["test"][field])
        rob = lambda spec: values(runs, lambda r: r["robustness"][spec]["mean_auc_held_out"])
        t2.append([label(k, keys), str(len(runs)), fmt(test("mean_auc_seen")), fmt(test("mean_auc_near")),
                   fmt(test("mean_auc_far")), fmt(test("mean_auc_held_out")), fmt(test("seen_minus_held_out_auc")),
                   fmt(test("mean_acc"))] + [fmt(rob(f"jpeg:{q}")) for q in (95, 75, 50)])

    per_gen = [[TEST_GENERATORS[g]] + [fmt(values(groups[k], lambda r: r["test"]["per_generator_auc"][g]))
                                      for k in keys] for g in TEST_GENERATORS]

    specs = [s for s in ROBUSTNESS if any(s in r.get("robustness", {}) for k in keys for r in groups[k])]
    rob_rows = [[label(k, keys)] + [fmt(values(groups[k], lambda r: r["robustness"][s]["mean_auc"])) for s in specs]
                for k in keys]

    names = [label(k, keys) for k in keys]
    lines = [f"## Table 2: test AUC by generator group ({MAIN_PREP}, mean ± std over seeds)", ""]
    lines += table(["Model", "Seeds", "Seen", "Near", "Far", "Held-out", "Gap", "Acc",
                    "Held-out JPEG 95", "JPEG 75", "JPEG 50"], t2)
    lines += ["## Test AUC by generator", ""] + table(["Generator"] + names, per_gen)
    if specs:
        lines += ["## Robustness: mean AUC over all 8 generators", ""] + table(["Model"] + specs, rob_rows)
    return lines, keys


def ablation_table(groups):
    n_preps = Counter((k[0], k[1], k[3]) for k in groups)
    keys = sorted((k for k in groups if n_preps[k[0], k[1], k[3]] > 1), key=model_order)
    if not keys:
        return []
    rows = []
    for k in keys:
        runs = groups[k]
        rows.append([label(k, keys), k[2], str(len(runs)),
                     fmt(values(runs, lambda r: r["test"]["mean_auc_held_out"])),
                     fmt(values(runs, lambda r: r["aligned_test"]["mean_auc_held_out"])),
                     fmt(values(runs, lambda r: r["test"]["mean_auc"])),
                     fmt(values(runs, lambda r: r["robustness"]["jpeg:96"]["mean_auc"])),
                     fmt(values(runs, lambda r: r["aligned_test"]["mean_auc"])),
                     fmt(values(runs, lambda r: mean_fake_acc(r["test"]), 100), 1),
                     fmt(values(runs, lambda r: mean_fake_acc(r["robustness"]["jpeg:96"]), 100), 1),
                     fmt(values(runs, lambda r: mean_fake_acc(r["aligned_test"]), 100), 1)])
    header = ["Model", "Prep", "Seeds", "Held-out AUC clean", "Held-out AUC aligned",
              "AUC all-8 clean", "AUC all-8 q=96", "AUC all-8 aligned",
              "Fake acc % clean", "Fake acc % q=96", "Fake acc % aligned"]
    return ["## Table 3: preprocessing ablation (mean ± std over seeds)", "",
            "q=96: real and fake JPEG-compressed at test time. Aligned: only non-JPEG fakes re-encoded at q96, "
            "so both classes are single-compressed JPEGs.", ""] + table(header, rows)


def figure(groups, keys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = list(TEST_GENERATORS)
    x = np.arange(len(gens))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, k in enumerate(keys):
        auc = [values(groups[k], lambda r: r["test"]["per_generator_auc"][g]) for g in gens]
        mean = [a.mean() if a.size else np.nan for a in auc]
        std = [a.std(ddof=1) if a.size > 1 else 0.0 for a in auc]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, mean, width, yerr=std, capsize=2,
               label=f"{label(k, keys)} (n={len(groups[k])})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, [TEST_GENERATORS[g] for g in gens])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC")
    ax.legend(ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png, .pdf")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", default=["runs_final/*/"], help="run dirs or quoted glob patterns")
    p.add_argument("--out", default="results/final_tables.md", help="markdown output ('' to only print)")
    p.add_argument("--fig", default="results/figures/fig1_auc_per_generator",
                   help="Figure 1 path without extension ('' to skip)")
    return p.parse_args()


def main():
    args = parse_args()
    groups = load_groups(args.runs)
    main_lines, main_keys = main_tables(groups)
    lines = ["# Final results", ""] + main_lines + ablation_table(groups)
    lines += ["Runs:", ""] + [f"- `{r['run_name']}`" for k in sorted(groups, key=model_order) for r in groups[k]]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    if args.fig and main_keys:
        figure(groups, main_keys, Path(args.fig))


if __name__ == "__main__":
    main()
