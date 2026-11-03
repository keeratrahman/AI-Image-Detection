import json

def fake_acc(d):
    v = d["per_generator_fake_acc"].values()
    return sum(v) / len(v)

for model, lr in (("resnet50", "3e-4"), ("deit-s", "3e-5")):
    print(model)
    for prep in ("raw-resize", "raw-crop", "jpeg-resize", "jpeg-crop"):
        m = json.load(open(f"runs/A_{model}_in1k_{prep}_lr{lr}_s0/metrics.json"))
        t, r = m["test"], m["robustness"]["jpeg:96"]
        # print(f"  {prep:<12} held-out AUC={t['mean_auc_held_out']:.4f}  "
        #       f"fake acc clean={fake_acc(t):.3f} jpeg96={fake_acc(r):.3f} drop={fake_acc(t)-fake_acc(r):.3f}")
        print(f"  {prep:<12} held-out AUC={t['mean_auc_held_out']:.4f}  "
              f"all-8 AUC clean={t['mean_auc']:.4f} jpeg96={r['mean_auc']:.4f}  "
              f"fake acc clean={fake_acc(t):.3f} jpeg96={fake_acc(r):.3f}")