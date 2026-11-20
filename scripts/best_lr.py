"""Print the LR sweep per model and mark the best LR (seen-generator validation only).
    python scripts/best_lr.py [runs_dir] [--tol 0.0005]

Selection rule: highest val AUC; any LR within --tol of that AUC counts as tied, and ties
are broken by lowest val loss. Both numbers come from the checkpointed (best-AUC) epoch,
i.e. metrics.json["val"]. A winner at the smallest or largest LR swept is flagged as an edge.
"""
import argparse
import glob
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir", nargs="?", default="runs")
ap.add_argument("--tol", type=float, default=0.0005)
args = ap.parse_args()

groups = defaultdict(list)
for f in glob.glob(f"{args.runs_dir}/*/metrics.json"):
    m = json.load(open(f))
    groups[(m["protocol"], m["model"], m["prep"])].append(m)

for key in sorted(groups):
    runs = sorted(groups[key], key=lambda m: (-m["lr"], m["seed"]))
    top_auc = max(m["val"]["auc"] for m in runs)
    tied = [m for m in runs if top_auc - m["val"]["auc"] <= args.tol]
    best = min(tied, key=lambda m: m["val"]["loss"])

    lrs = sorted({m["lr"] for m in runs})
    print(f"\n{key[0]} / {key[1]} / {key[2]}")
    for m in runs:
        mark = ""
        if m is best:
            mark = f"  <-- best: {args.runs_dir}/{m['run_name']}"
            if len(lrs) > 1 and m["lr"] in (lrs[0], lrs[-1]):
                mark += "  [EDGE of grid]"
        elif m in tied:
            mark = "  (tied on AUC)"
        print(f"  lr={m['lr']:<8g} seed={m['seed']}  val_auc={m['val']['auc']:.6f}  "
              f"val_loss={m['val']['loss']:.4f}  best_epoch={m['best_epoch']}{mark}")
    if len({m["seed"] for m in runs}) > 1:
        print("  note: several seeds in this group; this ranks individual runs, not LRs")
