import json, sys

runs = {p.rstrip("/").split("/")[-1]: json.load(open(p.rstrip("/") + "/metrics.json")) for p in sys.argv[1:]}
names = list(runs)
f = lambda x: f"{x:>12.4f}" if isinstance(x, (int, float)) else f"{'n/a':>12}"

print(f"{'':<26}" + "".join(f"{n.split('_')[1]:>12}" for n in names))
for g in runs[names[0]]["test"]["per_generator_auc"]:
    print(f"{g:<26}" + "".join(f(runs[n]["test"]["per_generator_auc"][g]) for n in names))
for k in ("mean_auc_seen", "mean_auc_near", "mean_auc_far", "mean_auc_held_out",
          "seen_minus_held_out_auc", "mean_acc"):
    print(f"{k:<26}" + "".join(f(runs[n]["test"].get(k)) for n in names))
print()
for spec in runs[names[0]].get("robustness", {}):
    print(f"{'mean_auc @ ' + spec:<26}" + "".join(f(runs[n]["robustness"][spec]["mean_auc"]) for n in names))