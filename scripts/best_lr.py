"""Print the LR sweep per model and mark the best LR (seen-generator validation AUC only).
    python scripts/best_lr.py [runs_dir]
"""
import glob
import json
import sys
from collections import defaultdict

runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
groups = defaultdict(list)
for f in glob.glob(f"{runs_dir}/*/metrics.json"):
    m = json.load(open(f))
    groups[(m["protocol"], m["model"], m["prep"])].append(m)

for key in sorted(groups):
    runs = sorted(groups[key], key=lambda m: -m["lr"])
    best = max(runs, key=lambda m: m["val"]["auc"])
    print(f"\n{key[0]} / {key[1]} / {key[2]}")
    for m in runs:
        mark = "  <-- best" if m is best else ""
        print(f"  lr={m['lr']:<8g} val_auc={m['val']['auc']:.4f}  best_epoch={m['best_epoch']}{mark}")