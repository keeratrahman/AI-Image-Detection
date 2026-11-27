#!/usr/bin/env python3
"""Task 6, step 1: extract frozen CLIP ViT-L/14 image features once (Ojha et al., 2023 baseline).

    python scripts/clip_features.py --manifest-dir ../genimage_subset/manifests
    python scripts/clip_features.py --manifest-dir ... --conditions aligned jpeg:96      # optional extras

Writes features/<model>/<condition>/<split>.npz with
    feats  [n, d] float32 (d = 768 for ViT-L/14; the CLIP image embedding, not L2-normalized)
    labels [n], paths [n] (relative to the dataset root, as in eval.py --save-preds), manifest

Preprocessing is our jpeg-crop eval view (upsample if short side < 224, 224 center crop, no flip)
with CLIP's mean/std instead of ImageNet's. Center crop is used for the training features too, so
features are extracted once.
  * clean:   A_train and A_val with fakes JPEG-aligned (as in jpeg-crop training); the 8 test
             sets untouched (jpeg_aligned=False, as in eval.py).
  * aligned: the 8 test sets with non-JPEG fakes re-encoded at q96 (eval.py --aligned-test).
  * <spec>:  the 8 test sets with a perturbation (jpeg:75, blur:2, rescale:0.5, ...), as in
             eval.py --robustness.
Finished files are skipped, so the script can be stopped (Ctrl+C) and rerun to continue.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.genimage import IMAGE_SIZE, GenImageDataset, MinShortSide  # noqa: E402
from aidet.naming import TEST_GENERATORS  # noqa: E402
from eval import condition_name, make_perturbation, pick_device  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TRAIN_SPLITS = ["A_train", "A_val"]


def clip_eval_transform():
    """jpeg-crop eval view with CLIP normalization."""
    return T.Compose([MinShortSide(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE), T.ToTensor(),
                      T.Normalize(CLIP_MEAN, CLIP_STD)])


def make_dataset(manifest, jpeg_aligned, perturbation=None, limit=None):
    ds = GenImageDataset(manifest, prep="jpeg-crop", train=False, jpeg_aligned=jpeg_aligned,
                         extra_transform=perturbation)
    ds.transform = clip_eval_transform()  # same view as the CNN/ViT runs, CLIP's mean/std
    if limit:
        ds.records = ds.records[:limit // 2] + ds.records[-(limit - limit // 2):]  # keep both classes
    return ds


@torch.no_grad()
def extract(model, ds, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, labels = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        feats.append(model.encode_image(x.to(device)).float().cpu())
        labels.append(y.float())
        if (i + 1) % 20 == 0:
            done = (i + 1) * batch_size
            print(f"    {done}/{len(ds)} images  ({done / (time.time() - t0):.0f} img/s)", flush=True)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def jobs(manifest_dir, conditions):
    """(condition, split, manifest path, jpeg_aligned, perturbation spec) for everything to extract."""
    out = []
    for cond in conditions:
        if cond == "clean":
            out += [("clean", s, manifest_dir / f"{s}.jsonl", True, None) for s in TRAIN_SPLITS]
            out += [("clean", g, manifest_dir / f"{g}.jsonl", False, None) for g in TEST_GENERATORS]
        elif cond == "aligned":
            out += [("aligned", g, manifest_dir / f"{g}.jsonl", True, None) for g in TEST_GENERATORS]
        else:
            make_perturbation(cond)  # validate the spec early
            out += [(condition_name(cond), g, manifest_dir / f"{g}.jsonl", False, cond) for g in TEST_GENERATORS]
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True, help="folder with A_train.jsonl, A_val.jsonl, test_*.jsonl")
    p.add_argument("--conditions", nargs="+", default=["clean"],
                   help="clean (train/val/test), aligned, and/or perturbation specs like jpeg:96")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai", help="open_clip pretrained tag ('none' = random init, for testing)")
    p.add_argument("--out-dir", default="features", help="features/<model-tag>/<condition>/<split>.npz")
    p.add_argument("--tag", default="clip-l14", help="subfolder name under --out-dir")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    p.add_argument("--limit", type=int, help="only the first/last N/2 images per split (quick test)")
    return p.parse_args()


def main():
    args = parse_args()
    import open_clip

    manifest_dir = Path(args.manifest_dir)
    todo = jobs(manifest_dir, args.conditions)
    missing = sorted({str(m) for _, _, m, _, _ in todo if not m.exists()})
    if missing:
        raise SystemExit("Missing manifests:\n  " + "\n  ".join(missing))

    out_root = Path(args.out_dir) / args.tag
    todo = [j for j in todo if not (out_root / j[0] / f"{j[1]}.npz").exists()]
    if not todo:
        print(f"Everything already extracted under {out_root}")
        return

    device = pick_device(args.device)
    pretrained = None if args.pretrained == "none" else args.pretrained
    model = open_clip.create_model(args.model, pretrained=pretrained).to(device).eval()
    print(f"{args.model} ({args.pretrained}) on {device}; {len(todo)} split(s) to extract -> {out_root}")

    for cond, split, manifest, aligned, spec in todo:
        pert = make_perturbation(spec) if spec else None
        ds = make_dataset(manifest, aligned, pert, args.limit)
        print(f"  [{cond}] {split}: {len(ds)} images (jpeg_aligned={aligned}, perturbation={spec})", flush=True)
        feats, labels = extract(model, ds, device, args.batch_size, args.num_workers)
        root = manifest.resolve().parent.parent
        paths = [os.path.relpath(r["path"], root) for r in ds.records]
        dest = out_root / cond / f"{split}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".tmp.npz")  # write then rename, so a Ctrl+C never leaves a half file
        np.savez(tmp, feats=feats, labels=labels, paths=np.array(paths), manifest=np.array(str(manifest)))
        tmp.rename(dest)
        print(f"    saved {dest}  feats {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
