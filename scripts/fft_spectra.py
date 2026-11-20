#!/usr/bin/env python3
"""Average Fourier spectra of high-pass noise residuals, real vs. fake, per test generator (Task 7a).

    python scripts/fft_spectra.py --manifest-dir ~/scratch/genimage_subset/manifests
    python scripts/fft_spectra.py --manifest-dir ... --jpeg-align --out results/figures/fft_spectra_aligned

Each image goes through the model's eval view (upsample if short side < 224, 224 center crop),
then grayscale. Residual = image - 3x3 median filter. The figure shows log10 of the mean power
spectrum for fakes, reals, and their difference (fake - real), one column per generator.
Writes <out>.png, <out>.pdf and <out>.npz.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidet.genimage import IMAGE_SIZE, MinShortSide, jpeg_reencode, read_manifest  # noqa: E402
from aidet.naming import TEST_GENERATORS  # noqa: E402

EVAL_VIEW = T.Compose([MinShortSide(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE)])
CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()  # Slurm allocation, not the whole node


def residual_power(path, jpeg_align):
    with Image.open(path) as im:
        is_jpeg = im.format == "JPEG"
        img = im.convert("RGB")
    if jpeg_align and not is_jpeg:
        img = jpeg_reencode(img)
    gray = EVAL_VIEW(img).convert("L")
    res = np.asarray(gray, np.float32) - np.asarray(gray.filter(ImageFilter.MedianFilter(3)), np.float32)
    return np.abs(np.fft.fft2(res)) ** 2


def chunk_power(paths, jpeg_align):
    total = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for p in paths:
        total += residual_power(p, jpeg_align)
    return total


def mean_log_spectrum(pool, paths, workers, jpeg_align):
    chunks = [c for c in np.array_split(paths, workers) if len(c)]
    total = sum(pool.map(partial(chunk_power, jpeg_align=jpeg_align), chunks))
    return np.log10(np.fft.fftshift(total / len(paths)) + 1e-12)


def plot(spectra, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = list(spectra)
    fig, axes = plt.subplots(3, len(gens), figsize=(2 * len(gens), 6.6), squeeze=False)
    both = np.stack([spectra[g][c] for g in gens for c in ("fake", "real")])
    vmin, vmax = np.percentile(both, [1, 99])
    diff = {g: spectra[g]["fake"] - spectra[g]["real"] for g in gens}
    lim = np.percentile(np.abs(np.stack(list(diff.values()))), 99)
    rows = [("Fake", lambda g: spectra[g]["fake"], dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("Real", lambda g: spectra[g]["real"], dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("Fake - real", lambda g: diff[g], dict(cmap="RdBu_r", vmin=-lim, vmax=lim))]
    for r, (name, get, style) in enumerate(rows):
        for c, g in enumerate(gens):
            ax = axes[r, c]
            im = ax.imshow(get(g), **style)
            ax.set(xticks=[], yticks=[])
            if r == 0:
                ax.set_title(TEST_GENERATORS[g])
        axes[r, 0].set_ylabel(name)
        fig.colorbar(im, ax=axes[r].tolist(), fraction=0.012, pad=0.01, label="log10 power")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True, help="folder with the test_<generator>.jsonl manifests")
    p.add_argument("--n", type=int, default=500, help="images per class per generator")
    p.add_argument("--jpeg-align", action="store_true", help="re-encode non-JPEG fakes at q96 first (as in training)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=CPUS)
    p.add_argument("--out", default="results/figures/fft_spectra", help="output path without extension")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spectra = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for gen in TEST_GENERATORS:
            records = read_manifest(Path(args.manifest_dir) / f"{gen}.jsonl")
            spectra[gen] = {}
            for cls, label in (("fake", 1), ("real", 0)):
                paths = [r["path"] for r in records if r["label"] == label]
                paths = rng.choice(paths, min(args.n, len(paths)), replace=False)
                spectra[gen][cls] = mean_log_spectrum(pool, paths, args.workers, args.jpeg_align)
            print(f"  {gen}: {len(paths)} images per class", flush=True)

    np.savez(out.with_suffix(".npz"), **{f"{g}_{c}": s for g, d in spectra.items() for c, s in d.items()})
    plot(spectra, out)
    print(f"Wrote {out}.png, .pdf, .npz")


if __name__ == "__main__":
    main()
