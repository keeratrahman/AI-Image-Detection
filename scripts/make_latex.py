#!/usr/bin/env python3
"""Write self-contained LaTeX tables/figures (pgfplots, data inlined) from runs/*/metrics.json.

    python scripts/make_latex.py          # writes figures/latex/*.tex

Upload figures/latex/ to the report and \\input{} each file. Re-run after new evals/seeds.
Preamble needs:  \\usepackage{booktabs,pgfplots}  \\pgfplotsset{compat=1.18}
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "figures" / "latex"

MODELS = [  # (key, label, run)
    ("resnet18", "ResNet-18 (baseline)", "A_resnet18_in1k_jpeg-crop_lr3e-4_s0"),
    ("resnet50", "ResNet-50", "A_resnet50_in1k_jpeg-crop_lr3e-4_s0"),
    ("deits", "DeiT-S", "A_deit-s_in1k_jpeg-crop_lr3e-5_s0"),
]
H4 = [("ResNet-50", "resnet50", "3e-4"), ("DeiT-S", "deit-s", "3e-5")]
PREPS = ["raw-resize", "raw-crop", "jpeg-resize", "jpeg-crop"]
GENS = [("sdv14", "SD1.4"), ("sdv15", "SD1.5"), ("wukong", "Wukong"), ("adm", "ADM"),
        ("glide", "GLIDE"), ("vqdm", "VQDM"), ("midjourney", "MJ"), ("biggan", "BigGAN")]
GROUPS = {"seen": ["sdv14"], "near": ["sdv15", "wukong"],
          "far": ["adm", "glide", "vqdm", "midjourney", "biggan"]}
GROUPS["held_out"] = GROUPS["near"] + GROUPS["far"]
JPEG_Q = [100, 95, 75, 50]  # 100 = clean; q=96 skipped so every model has every point


def load(run):
    return json.load(open(RUNS / run / "metrics.json"))


def gen(section, key, g):
    d = section[f"per_generator_{key}"]
    return d.get(g, d.get(f"test_{g}"))


def avg(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def group_auc(s, grp):
    return avg([gen(s, "auc", g) for g in GROUPS[grp]])


def fake_acc(s):
    return avg(list(s["per_generator_fake_acc"].values()))


def f3(x):
    return "--" if x is None or math.isnan(x) else f"{x:.3f}"


def bold_best(vals, higher=True):
    ok = [v for v in vals if not math.isnan(v)]
    best = max(ok) if higher else min(ok)
    return [f"\\textbf{{{f3(v)}}}" if v == best else f3(v) for v in vals]


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")
    print(f"wrote figures/latex/{name}")


def main_table(ms):
    cols = {
        "seen": [group_auc(m["test"], "seen") for m in ms],
        "near": [group_auc(m["test"], "near") for m in ms],
        "far": [group_auc(m["test"], "far") for m in ms],
        "held": [group_auc(m["test"], "held_out") for m in ms],
    }
    cols["gap"] = [s - h for s, h in zip(cols["seen"], cols["held"])]
    cols["acc"] = [m["test"]["mean_acc"] for m in ms]
    # held-out AUC under JPEG compression (q=96 skipped: ResNet-18 was not evaluated at it)
    jpeg_qs = [q for q in JPEG_Q if q != 100]
    for q in jpeg_qs:
        cols[f"q{q}"] = [group_auc(m["robustness"][f"jpeg:{q}"], "held_out") for m in ms]
    order = ["seen", "near", "far", "held", "gap", "acc"] + [f"q{q}" for q in jpeg_qs]
    fmt = {k: bold_best(v, higher=(k != "gap")) for k, v in cols.items()}
    rows = "\n".join(
        f"{label} & " + " & ".join(fmt[k][i] for k in order) + r" \\"
        for i, (_, label, _) in enumerate(MODELS))
    nq = len(jpeg_qs)
    write("tab_main.tex", rf"""
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l{"c" * (6 + nq)}}}
\toprule
 & \multicolumn{{6}}{{c}}{{Clean test images}} & \multicolumn{{{nq}}}{{c}}{{Held-out, JPEG $q$}} \\
\cmidrule(lr){{2-7}} \cmidrule(lr){{8-{7 + nq}}}
Model & Seen & Near & Far & Held-out & Gap & Acc & {" & ".join(str(q) for q in jpeg_qs)} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{Test results by generator group (seed 0). Last {nq} columns: held-out AUC after JPEG
compression at quality $q$.}}
\label{{tab:main}}
\end{{table}}
""")


def h4_table():
    lines = []
    for label, key, lr in H4:
        for j, prep in enumerate(PREPS):
            m = load(f"A_{key}_in1k_{prep}_lr{lr}_s0")
            t, r = m["test"], m["robustness"]["jpeg:96"]
            fc, fr = fake_acc(t) * 100, fake_acc(r) * 100
            name = rf"\multirow{{4}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(f"{name} & \\texttt{{{prep}}} & {f3(group_auc(t, 'held_out'))} & "
                         f"{f3(t['mean_auc'])} & {f3(r['mean_auc'])} & "
                         f"{fc:.1f} & {fr:.1f} & ${fc - fr:+.1f}$ \\\\")
        lines.append(r"\midrule")
    rows = "\n".join(lines[:-1])
    write("tab_h4.tex", rf"""
\begin{{table}}[t]
\centering
\caption{{Preprocessing ablation (H4, seed 0); $\Delta$ = clean $-$ $q{{=}}96$.}}
\label{{tab:h4}}
\small
\begin{{tabular}}{{llcccccc}}
\toprule
 & & Held-out & \multicolumn{{2}}{{c}}{{AUC (all 8)}} & \multicolumn{{3}}{{c}}{{Fake acc.\ (\%)}} \\
\cmidrule(lr){{4-5}} \cmidrule(lr){{6-8}}
Model & Prep & AUC & Clean & $q{{=}}96$ & Clean & $q{{=}}96$ & $\Delta$ \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def per_generator_fig(ms):
    data = "\n".join(f"{lab} " + " ".join(f"{gen(m['test'], 'auc', g):.4f}" for m in ms)
                     for g, lab in GENS)
    keys = " ".join(k for k, _, _ in MODELS)
    plots = "\n".join(rf"\addplot table[x=gen, y={k}] {{\pergen}};" for k, _, _ in MODELS)
    legend = ",".join(l.replace(" (baseline)", "") for _, l, _ in MODELS)
    write("fig_per_generator.tex", rf"""
\pgfplotstableread{{
gen {keys}
{data}
}}\pergen
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  ybar, bar width=4pt, width=\linewidth, height=4cm,
  ymin=0.4, ymax=1.02, ylabel={{AUC}},
  symbolic x coords={{{",".join(l for _, l in GENS)}}}, xtick=data,
  x tick label style={{font=\footnotesize}}, enlarge x limits=0.08,
  legend style={{at={{(0.5,1.02)}}, anchor=south, legend columns=-1, font=\footnotesize, draw=none}},
  ymajorgrids, grid style={{gray!25}},
  extra y ticks={{0.5}}, extra y tick style={{grid=major, grid style={{dashed, black!60}}}},
]
{plots}
\legend{{{legend}}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Clean test AUC per generator (seed 0); dashed line: chance.}}
\label{{fig:per-generator}}
\end{{figure}}
""")


def jpeg_fig(ms):
    def sec(m, q):
        return m["test"] if q == 100 else m["robustness"].get(f"jpeg:{q}")
    rows = []
    for q in JPEG_Q:
        vals = []
        for m in ms:
            s = sec(m, q)
            vals.append("nan" if s is None else f"{group_auc(s, 'held_out'):.4f}")
        rows.append(f"{q} " + " ".join(vals))
    keys = " ".join(k for k, _, _ in MODELS)
    marks = ["square*", "*", "triangle*"]
    plots = "\n".join(rf"\addplot+[mark={mk}] table[x=q, y={k}] {{\jpegq}};"
                      for (k, _, _), mk in zip(MODELS, marks))
    legend = ",".join(l.replace(" (baseline)", "") for _, l, _ in MODELS)
    write("fig_jpeg.tex", rf"""
\pgfplotstableread{{
q {keys}
{chr(10).join(rows)}
}}\jpegq
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.8\linewidth, height=4cm, x dir=reverse,
  xlabel={{JPEG quality $q$ (100 = no compression)}}, ylabel={{Held-out AUC}},
  xtick={{{",".join(str(q) for q in JPEG_Q)}}}, ymin=0.55, ymax=0.9,
  legend pos=south west, legend style={{font=\footnotesize}},
  ymajorgrids, grid style={{gray!25}}, unbounded coords=jump,
]
{plots}
\legend{{{legend}}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Held-out AUC under JPEG compression at quality $q$ (seed 0).}}
\label{{fig:jpeg}}
\end{{figure}}
""")


def main():
    ms = [load(run) for _, _, run in MODELS]
    main_table(ms)
    h4_table()
    per_generator_fig(ms)
    jpeg_fig(ms)


if __name__ == "__main__":
    main()
