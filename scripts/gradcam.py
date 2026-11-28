#!/usr/bin/env python3
"""Task 7b: Grad-CAM maps for ResNet-50 and DeiT-S on chosen fake test images.

    python scripts/gradcam.py --manifest-dir ../genimage_subset/manifests

Picks fake images from the saved clean predictions (eval.py --save-preds) of both models:
  * SD v1.4, both right                      (most confident for both: highest min logit)
  * GLIDE,   DeiT-S right, ResNet-50 wrong   (largest DeiT-S minus ResNet-50 logit)
  * BigGAN,  both wrong                      (most confidently wrong: lowest max logit)
and draws one row per image: the image (eval view), the ResNet-50 map, the DeiT-S map.

Maps show evidence for the model's single "fake" logit (Grad-CAM target = that logit).
  * ResNet-50: Grad-CAM at layer4[-1] (7x7).
  * DeiT-S:    Grad-CAM at blocks[-1].norm1, patch tokens reshaped to 14x14 (CLS token dropped).
Images go through the same eval view as eval.py (upsample if short side < 224, 224 center crop,
ImageNet normalization); the script re-predicts each image and checks it against the saved logit.

Writes <out>.png, <out>.pdf and <out>_examples.json (which images, their logits).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.genimage import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MinShortSide  # noqa: E402
from aidet.naming import TEST_GENERATORS  # noqa: E402
from eval import load_run  # noqa: E402

# (generator, title, selection rule on fake images; r = ResNet-50 logit, d = DeiT-S logit)
CATEGORIES = [
    ("test_sdv14", "both right", lambda r, d: (r > 0) & (d > 0), lambda r, d: -np.minimum(r, d)),
    ("test_glide", "DeiT-S right, ResNet-50 wrong", lambda r, d: (d > 0) & (r <= 0), lambda r, d: -(d - r)),
    ("test_biggan", "both wrong", lambda r, d: (r <= 0) & (d <= 0), lambda r, d: np.maximum(r, d)),
]
VIEW = T.Compose([MinShortSide(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE)])
TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_preds(run_dir, gen):
    d = np.load(Path(run_dir) / "preds" / "clean" / f"{gen}.npz")
    return d["logits"], d["labels"], d["paths"]


def vit_reshape(prefix_tokens, grid=IMAGE_SIZE // 16):
    def reshape(t):
        t = t[:, prefix_tokens:, :]                       # drop CLS (and distillation) tokens
        return t.reshape(t.size(0), grid, grid, t.size(2)).permute(0, 3, 1, 2)
    return reshape


def pick_examples(resnet_dir, deit_dir, per_category):
    picks = []
    for gen, title, keep, rank in CATEGORIES:
        r, y, paths = load_preds(resnet_dir, gen)
        d, y2, paths2 = load_preds(deit_dir, gen)
        if not ((paths == paths2).all() and (y == y2).all()):
            raise SystemExit(f"{gen}: ResNet-50 and DeiT-S preds are not on the same images in the same order")
        idx = np.flatnonzero((y > 0.5) & keep(r, d))
        if len(idx) == 0:
            print(f"  warning: no fake images in {gen} match '{title}'")
            continue
        idx = idx[np.argsort(rank(r[idx], d[idx]), kind="stable")][:per_category]
        print(f"  {gen}: {title}: {np.sum((y > 0.5) & keep(r, d))} candidates, using rows {idx.tolist()}")
        picks += [{"generator": gen, "category": title, "row": int(i), "path": str(paths[i]),
                   "resnet50_logit": float(r[i]), "deit_s_logit": float(d[i])} for i in idx]
    return picks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resnet", default="runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s0")
    p.add_argument("--deit", default="runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0")
    p.add_argument("--manifest-dir", required=True, help="genimage_subset/manifests (images are under its parent)")
    p.add_argument("--per-category", type=int, default=4, help="images per category (3 categories)")
    p.add_argument("--out", default="results/figures/gradcam", help="output path without extension")
    p.add_argument("--device", default="cpu", help="cpu is fine (a handful of images) and avoids MPS backward quirks")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    device = torch.device(args.device)
    resnet, rcfg, _ = load_run(args.resnet, device)
    deit, dcfg, _ = load_run(args.deit, device)
    for name, cfg in (("ResNet-50", rcfg), ("DeiT-S", dcfg)):
        if not cfg["prep"].endswith("crop"):
            raise SystemExit(f"{name} run uses prep {cfg['prep']}; this script assumes the crop eval view")
    print(f"ResNet-50: {rcfg['model']} lr={rcfg['lr']}   DeiT-S: {dcfg['model']} lr={dcfg['lr']}")

    picks = pick_examples(args.resnet, args.deit, args.per_category)
    if not picks:
        raise SystemExit("No examples found")
    data_root = Path(args.manifest_dir).resolve().parent

    cams = {
        "ResNet-50": GradCAM(model=resnet, target_layers=[resnet.layer4[-1]]),
        "DeiT-S": GradCAM(model=deit, target_layers=[deit.blocks[-1].norm1],
                          reshape_transform=vit_reshape(getattr(deit, "num_prefix_tokens", 1))),
    }
    models = {"ResNet-50": (resnet, "resnet50_logit"), "DeiT-S": (deit, "deit_s_logit")}
    target = [ClassifierOutputTarget(0)]  # the single fake logit

    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 2.5 * len(picks)), squeeze=False)
    for row, pk in enumerate(picks):
        with Image.open(data_root / pk["path"]) as im:
            view = VIEW(im.convert("RGB"))
        x = TO_TENSOR(view).unsqueeze(0).to(device)
        rgb = np.asarray(view, dtype=np.float32) / 255.0
        axes[row, 0].imshow(rgb)
        gen_label = TEST_GENERATORS[pk["generator"]]
        axes[row, 0].set_ylabel(f"{gen_label}\n{pk['category']}", fontsize=8)
        for col, name in enumerate(["ResNet-50", "DeiT-S"], start=1):
            model, key = models[name]
            with torch.no_grad():
                logit = float(model(x).reshape(-1)[0])
            if abs(logit - pk[key]) > 1e-2 * max(1.0, abs(pk[key])):
                print(f"  warning: {name} logit {logit:.3f} != saved {pk[key]:.3f} for {pk['path']}")
            heat = cams[name](input_tensor=x, targets=target)[0]
            axes[row, col].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
            axes[row, col].set_title(f"{name}: p(fake)={sigmoid(logit):.2f}", fontsize=8)
        print(f"  row {row}: {gen_label:<6} {pk['category']:<30} R50 p={sigmoid(pk['resnet50_logit']):.2f} "
              f"DeiT p={sigmoid(pk['deit_s_logit']):.2f}  {pk['path']}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
    axes[0, 0].set_title("image (fake, 224 center crop)", fontsize=8)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    info = {"resnet50_run": args.resnet, "deit_s_run": args.deit, "method": {
        "ResNet-50": "Grad-CAM at layer4[-1]", "DeiT-S": "Grad-CAM at blocks[-1].norm1, patch tokens -> 14x14",
        "target": "the single fake logit"}, "examples": picks}
    Path(f"{out}_examples.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {out}.png, {out}.pdf, {out}_examples.json")


if __name__ == "__main__":
    main()
