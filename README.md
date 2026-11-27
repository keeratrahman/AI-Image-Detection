# AI-Image-Detection

Does architecture affect how well an AI-generated-image detector generalizes to generators it never saw?

We train binary real-vs-fake classifiers on **one** generator, Stable Diffusion v1.4 (from GenImage), and
test them on **8** generators:

| Group | Generators | Notes |
|---|---|---|
| Seen | SD v1.4 | the training generator |
| Near | SD v1.5, Wukong | share SD v1.4's latent-diffusion decoder |
| Far | ADM, GLIDE, VQDM, Midjourney, BigGAN | different architectures; BigGAN is a GAN |

The main comparison is a parameter-matched CNN vs. ViT:
- **ResNet-50** (25.6M parameters)
- **DeiT-S** (22M)

Both use ImageNet-1k init and the same data, schedule and seeds. **ResNet-18** is a small CNN baseline, and a
frozen **CLIP ViT-L/14 linear probe** (Ojha et al., 2023) is the prior-method reference.

> `docs/PLAN.md` is the original (midterm) plan and is out of date. The final work followed the team's
> "Tasks - Final Report" list (Tasks 1–8). The [Final results](#final-results) and
> [What each task produced](#what-each-task-produced) sections below describe what was actually done.

## Final results

Full tables: [`results/final_tables.md`](results/final_tables.md). Figure 1:
[`results/figures/fig1_auc_per_generator.png`](results/figures/fig1_auc_per_generator.png). Every number
comes from a `metrics.json` under `runs_final/`.

Clean test AUC (mean ± std over 3 seeds; CLIP is 1 deterministic run):

| Model | Seen | Near | Far | Held-out | Seen − held-out |
|---|---|---|---|---|---|
| ResNet-18 | 0.998 | 0.982 | 0.716 ± 0.008 | 0.792 ± 0.006 | 0.206 |
| ResNet-50 | 0.999 | 0.986 | 0.713 ± 0.044 | 0.791 ± 0.031 | 0.208 |
| DeiT-S | 0.998 | 0.989 | 0.787 ± 0.008 | 0.845 ± 0.006 | 0.154 |
| CLIP ViT-L/14 probe | 0.989 | 0.974 | 0.828 | 0.870 | 0.120 |

### Takeaways

- **H1 is supported: the ViT generalizes better than the parameter-matched CNN.**
  - DeiT-S beats ResNet-50 on the far generators by **7.5 AUC points, 95% CI [6.7, 8.2]**. This is a paired
    bootstrap over test images, averaged over 3 seeds; the seed-to-seed std is 3.8 and the per-seed gaps are
    +7.7, +3.6 and +11.1.
  - This passes the pre-set rule (gap ≥ 3 points and CI excludes 0). Details:
    [`results/h1_ci.md`](results/h1_ci.md).
  - On the near generators the gap is about 0, as predicted.
  - The gap is driven by BigGAN (+26.5), GLIDE (+9.5) and VQDM (+3.4). ADM goes the other way (−2.4).
  - ResNet-50 is also much less stable across seeds on far generators (std 0.044 vs. 0.008).
- **H2 is partly supported: BigGAN is the hardest generator for the CNNs and CLIP, but not for DeiT-S.**
  - BigGAN AUC: ResNet-50 0.46, ResNet-18 0.51, CLIP 0.63.
  - DeiT-S scores 0.72 on BigGAN; its lowest generator is ADM (0.67).
- **The frozen CLIP probe generalizes best, but is the least robust.**
  - It has the highest held-out AUC and the smallest seen-versus-held-out gap, with large gains on ADM and
    VQDM.
  - It is still beaten by DeiT-S on BigGAN.
  - It falls to the bottom under JPEG q50 and blur.
  - See [`results/clip_probe.md`](results/clip_probe.md).
- **Grad-CAM (one figure, 12 images):** ResNet-50's evidence sits in a few large blobs, while DeiT-S's is
  spread over many patches.
  - On GLIDE images that only DeiT-S gets right, ResNet-50's map is nearly empty.
  - On BigGAN, ResNet-50's map is empty and DeiT-S's has only a few small spots.
  - See [`results/gradcam.md`](results/gradcam.md).
- **Preprocessing ablation (ResNet-50, seed 0 for the three non-default variants):**
  - Resizing the whole image to 224 hurts a lot compared with cropping (held-out AUC 0.70–0.71 vs.
    0.79–0.86).
  - On the clean test, `raw-crop` (no JPEG alignment) looks best. On the format-matched (aligned) test, where
    both classes are single-compressed JPEGs, `jpeg-crop` overtakes it (0.898 vs. 0.867). So part of
    `raw-crop`'s clean score comes from the "PNG = fake" shortcut (Grommelt et al., 2024).
  - See Table 3 in [`results/final_tables.md`](results/final_tables.md).
- **Limitations to report:**
  - BigGAN images (128 px) must be upsampled to 224.
  - The prep variants reuse the `jpeg-crop` LR.
  - There is one training generator, and the data is a 12k-image training subset.
  - Some best hyperparameters sit at the edge of their grids: DeiT-S LR 1e-5 and ResNet-18 LR 1e-3 in the
    seed-0 sweep (the final runs keep 3e-5 and 3e-4; see `bestruns.txt`), and the CLIP probe's C = 10.

## What each task produced

| Task | Owner | What | Output |
|---|---|---|---|
| 1 | Keerat | Seeds 0–2 for ResNet-50, DeiT-S, ResNet-18 (`jpeg-crop`, in1k init, best LR), trained locally on Apple MPS (ResNet-50 later retrained at LR 1e-4 on PACE, see Task 2) | `runs_final/A_{resnet50,deit-s,resnet18}_*_s{0,1,2}/` |
| 2 | Ahmed M / Ahmed H | LR grid extended at the edges (1e-3 for ResNets, 1e-5 for DeiT-S), re-run on PACE. ResNet-50 moved to LR 1e-4 and its seeds were retrained | `bestruns.txt`, `runs_pace/` |
| 3a | Keerat | `eval.py --save-preds`: per-image logits for every test condition | `<run>/preds/<condition>/<generator>.npz` |
| 3b | Keerat | `scripts/bootstrap_ci.py`: paired bootstrap CI for the H1 gap | `results/h1_ci.md` |
| 4 | Nhi | `eval.py --aligned-test`: format-matched test (non-JPEG fakes re-encoded at q96) | `aligned_test` block in `metrics.json` |
| 5 | Ahmed M / Ahmed H | Preprocessing-ablation runs (`raw-resize`, `raw-crop`, `jpeg-resize`) for ResNet-50 at LR 1e-4 | `runs_final/A_resnet50_in1k_{raw-resize,raw-crop,jpeg-resize}_lr1e-4_s0/` |
| 6 | Keerat (took over from Ahmed A) | CLIP ViT-L/14 linear probe: frozen features, logistic regression, same evaluation as every run | `runs_final/A_clip-l14-probe_jpeg-crop_s0/`, `results/clip_probe.md` |
| 7a | Nhi | Mean FFT spectra of noise residuals, real vs. fake, per generator | `results/figures/fft_spectra*.png` |
| 7b | Keerat (took over from Ahmed A) | Grad-CAM maps for ResNet-50 and DeiT-S on chosen test images | `results/figures/gradcam.png`, `results/gradcam.md` |
| — | (shared) | Mean ± std tables and Figure 1 over all `runs_final/` runs | `scripts/aggregate_seeds.py` → `results/final_tables.md`, `results/figures/fig1_*` |

### Task 1: seeds 0–2

- **Configs:** generated with `make_configs.py --output-dir runs_final --out configs/final`.
- **Runs:** each run was trained with `train.py` and evaluated with `eval.py` on all 8 test sets, with
  robustness specs `jpeg:95 jpeg:75 jpeg:50 jpeg:96 blur:1 blur:2 rescale:0.5` and `--save-preds`.
- **ResNet-50 retrain:** after Task 2 moved ResNet-50 to LR 1e-4, its three seeds were retrained at 1e-4 on
  PACE (CUDA). DeiT-S and ResNet-18 are the original MPS runs.
  - The `config.yaml` saved inside each `runs_final/` run is the exact config used.
  - `configs/final/` still holds the original `lr3e-4` ResNet-50 files.

### Task 2: LR grid extended at the edges

- **Why:** at the midterm, every model's best LR sat at the edge of its grid.
- **What was run:** the seed-0 sweep was re-run on PACE with one extra LR per model. Configs are in
  `configs/lr_edges/` and runs in `runs_pace/`.
  - ResNet-50 and ResNet-18: {1e-3, 3e-4, 1e-4, 3e-5}
  - DeiT-S: {3e-4, 1e-4, 3e-5, 1e-5}
- **Selection rule (fixed before looking):** highest seen-generator validation AUC. LRs within 0.0005 of the
  best count as tied, and ties are broken by lowest validation loss (`python scripts/best_lr.py runs_pace`).
- **Outcome** (full sweep in `bestruns.txt`):
  - **ResNet-50:** moved to **1e-4**, inside the grid, so its three seeds were retrained.
  - **ResNet-18 → 1e-3, DeiT-S → 1e-5:** both are again at the edge. Following the plan's rule, the grid
    was not extended further. The final runs keep the original LRs (ResNet-18 3e-4, DeiT-S 3e-5), and this
    is reported as a limitation.

### Task 3: per-image predictions and the H1 CI

- **`eval.py --save-preds`:** writes `logits`, `labels`, `paths` (relative to the dataset root) and the
  manifest path to `<run>/preds/<condition>/<generator>.npz`.
  - `<condition>` is `clean`, `aligned`, or the perturbation spec with `:` replaced by `_` (e.g. `jpeg_75`).
  - Test loaders do not shuffle, so row *i* is the same image in every run.
- **`scripts/bootstrap_ci.py`:**
  - Point estimate: AUC per run, averaged over seeds, then over the 5 far generators. Gap = DeiT-S −
    ResNet-50.
  - CI: 10,000 resamples, drawing real and fake images with replacement within each generator. The same
    indices are used for every model and seed. CI = 2.5th–97.5th percentile.
  - The script also checks that all runs scored the same images in the same order.

### Task 4: format-matched (aligned) test

- **Why:** in every test set the reals are JPEG and most fakes are PNG, so a model can score well with the
  rule "PNG = fake". The `jpeg:96` robustness test does not remove this, because it compresses the reals a
  second time and the fakes only once.
- **What:** `eval.py --aligned-test` builds the test sets with `jpeg_aligned=True`, which re-encodes only
  the non-JPEG fakes at q96. Both classes are then single-compressed JPEGs.
  - Results go to `metrics["aligned_test"]` (same fields as `test`; `test` is not overwritten).
  - Per-image preds go to `preds/aligned/`.
- **Coverage:**
  - Done for the ResNet-50 `jpeg-crop` runs (3 seeds), the three ResNet-50 ablation variants, and the
    CLIP probe.
  - Not yet run for DeiT-S or ResNet-18, so those cells are empty in the tables.
  - There is no separate `results/aligned_test.md`. The clean vs. aligned columns are in Table 3 of
    `results/final_tables.md`.
- **Caveat for the report:** the aligned test matches the `jpeg-*` training distribution, so it slightly
  favors those models. Report clean and aligned results side by side.

### Task 5: preprocessing-ablation runs

- **Variants:** the three non-default preprocessing variants, each trained at ResNet-50's final LR (1e-4).
  - `raw-resize`: no JPEG alignment, resize to 224
  - `raw-crop`: no JPEG alignment, crop
  - `jpeg-resize`: JPEG alignment, resize
- **Evaluation:** each run was evaluated with the clean test, `--robustness jpeg:96`, `--aligned-test` and
  `--save-preds`.
- **Scripts:** `scripts/pace/eval.sbatch` runs evals like these as a SLURM array over run folders.
- **Coverage:**
  - Only seed 0 for ResNet-50 exists: `runs_final/A_resnet50_in1k_{raw-resize,raw-crop,jpeg-resize}_lr1e-4_s0/`.
  - The `jpeg-crop` row of Table 3 reuses the 3 Task 1 seeds.
  - The plan's seeds 1–2 and the DeiT-S variants were not run, so the ablation rows are single-seed.
- **Caveat:** the variants reuse the `jpeg-crop` LR instead of their own sweep.

### Task 6: CLIP ViT-L/14 linear probe

Two steps, so the slow part (feature extraction) runs once:

1. `scripts/clip_features.py` extracts frozen open_clip `ViT-L-14` (`openai`) image embeddings (768-d).
   - Images go through our `jpeg-crop` eval view (224 center crop) with CLIP's mean/std.
   - Output: `features/clip-l14/<condition>/<split>.npz` (not in git).
     - `clean` = `A_train` and `A_val` with JPEG-aligned fakes, plus the 8 untouched test sets.
     - `aligned` and each perturbation spec are test sets only.
   - Finished files are skipped, so it can be stopped and resumed.
2. `scripts/clip_probe.py` fits a logistic regression on `A_train` and picks C ∈ {0.01, 0.1, 1, 10} by
   `A_val` AUC (same tie rule as the LR sweep).
   - It scores every test condition with `eval.score` / `eval.summarize`.
   - It writes `metrics.json` in the same format as trained runs, plus per-image preds. So `compare.py`,
     `aggregate_seeds.py` and `bootstrap_ci.py` work on it unchanged.

### Task 7a: FFT spectra

- **Script:** `scripts/fft_spectra.py`. It uses test images only, with no model involved.
- **Method:** for each generator, take 500 fakes and 500 reals, apply the same 224 center-crop eval view,
  and convert to grayscale.
  - High-pass residual = image minus its 3×3 median filter.
  - The figure shows log10 of the mean power spectrum (`fftshift(fft2(residual))`) for fakes, reals and
    fake − real, one column per generator.
- **Figures:**
  - `results/figures/fft_spectra.png`: test images as stored.
  - `results/figures/fft_spectra_aligned.png`: run with `--jpeg-align`, which re-encodes non-JPEG fakes at
    q96 first, as in training. This separates generator artifacts from the JPEG-vs-PNG difference.
- **What to look for:** periodic grid peaks in SD v1.4 / v1.5 / Wukong (from the shared VAE upsampler), and
  a different pattern for BigGAN. This would support H1's reasoning: the CNN fits SD-specific peaks that do
  not transfer.

### Task 7b: Grad-CAM

- **Script:** `scripts/gradcam.py` picks fake test images automatically from the two models' saved clean
  predictions:
  - SD v1.4 where both models are right
  - GLIDE where DeiT-S is right and ResNet-50 wrong
  - BigGAN where both are wrong
- **Maps:**
  - ResNet-50: Grad-CAM at `layer4[-1]`.
  - DeiT-S: Grad-CAM at `blocks[-1].norm1`, patch tokens reshaped to 14×14.
  - The target for both is the single fake logit.
- **Check:** the script re-predicts each image and checks the result against the saved logit.

## Layout

| Path | What it does |
|---|---|
| `aidet/data.py` | DataLoader construction; picks the stub or GenImage dataset from the config |
| `aidet/genimage.py` | GenImage dataset: reads a JSONL manifest, JPEG alignment, crop/resize to 224 |
| `aidet/models.py` | Model registry: `resnet18`, `resnet50`, `deit-s` (ImageNet-1k or scratch init) and `cnn-small` (scratch only), one output logit |
| `aidet/metrics.py` | ROC-AUC. Accuracy, precision, recall and F1 are computed in `eval.py` |
| `aidet/naming.py` | Run names from config values (`{protocol}_{model}_{init}_{prep}_lr{lr}_s{seed}`), test generator names |
| `train.py` | Training: `python train.py --config <config.yaml>`. Auto-selects CUDA, then Apple GPU (MPS), then CPU |
| `eval.py` | Evaluates a run on per-generator test manifests; clean, `--aligned-test`, `--robustness`, `--save-preds`. Writes into the run's `metrics.json` |
| `make_configs.py` | Generates experiment YAML configs |
| `compare.py` | Per-generator AUC, group means and robustness for several runs side by side |
| `ablation.py` | Midterm preprocessing-ablation table (reads `runs/`, seed 0) |
| `bench.py` | Times data loading and the model step |
| `scripts/aggregate_seeds.py` | Final tables (mean ± std over seeds) and Figure 1 from `runs_final/` |
| `scripts/bootstrap_ci.py` | Task 3b: paired bootstrap CI for the H1 gap |
| `scripts/clip_features.py` | Task 6 step 1: frozen CLIP ViT-L/14 features |
| `scripts/clip_probe.py` | Task 6 step 2: logistic-regression probe + evaluation |
| `scripts/gradcam.py` | Task 7b: Grad-CAM figure for ResNet-50 and DeiT-S |
| `scripts/fft_spectra.py` | Task 7a: FFT spectra of real vs. fake noise residuals |
| `scripts/best_lr.py` | Best LR per model by seen-generator validation AUC (ties within 0.0005 → lowest val loss) |
| `scripts/export_plot_data.py`, `scripts/make_latex.py` | Midterm CSV / LaTeX exports (read `runs/`) |
| `scripts/setup_local.sh` | One-time local setup: venv, pretrained weights, LR-sweep configs |
| `scripts/run_lr_sweep.sh` | Runs every config in `CONFIG_DIR` (default `configs/lr_sweep/`), skipping finished runs |
| `scripts/pace/` | PACE-ICE (SLURM) scripts: training array (`lr_sweep.sbatch`, takes `CONFIG_DIR`) and eval array (`eval.sbatch`) |
| `scripts/make_subset.py`, `scripts/sample_zip.py` | Build the balanced GenImage subset and its manifests (see `docs/DATA.md`) |
| `configs/` | `lr_sweep/` (midterm sweep + ablation), `lr_edges/` (Task 2 grid), `final/` (Task 1 seeds). `smoke/` is generated locally (see Setup) |
| `runs/` | Midterm runs (seed 0, M5) |
| `runs_pace/` | Task 2 LR sweep re-run on PACE |
| `runs_final/` | Final runs: 3 models × 3 seeds, the ablation variants, the CLIP probe |
| `results/` | `final_tables.md`, `h1_ci.md`, `clip_probe.md`, `gradcam.md`, midterm write-ups, `figures/` |
| `docs/DATA.md` | Download GenImage and build the subset |
| `docs/PLAN.md` | Original midterm plan (out of date; see note at the top) |

**What git tracks:** only `metrics.json` and `config.yaml` from each run folder. Checkpoints (`best.pt`),
per-image preds, CLIP features (`features/`), W&B files and the dataset are not tracked; they live on shared
storage.

## Setup

```
python -m venv .venv && source .venv/bin/activate      # or: uv venv; any Python 3.11+
pip install -r requirements.txt
```

Quick pipeline check on stub data (no dataset needed). `configs/smoke/` is not in git, so generate it first:

```
python make_configs.py --data stub --init scratch --models resnet18 --lrs 1e-3 \
  --preps jpeg-crop --epochs 3 --batch-size 32 --num-workers 0 --wandb-mode disabled \
  --output-dir runs_smoke --out configs/smoke
python train.py --config configs/smoke/A_resnet18_scratch_jpeg-crop_lr1e-3_s0.yaml
```

Data: follow [docs/DATA.md](docs/DATA.md). The commands below assume the subset is at `../genimage_subset`.

## Reproducing the final results

Set the manifest folder once:

```
M=../genimage_subset/manifests
```

**1. Train a run (Task 1).** Example: DeiT-S, seed 0.

```
python train.py --config configs/final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0.yaml
```

**2. Evaluate it.** Use only the `test_*` manifests, never `A_train` / `A_val`:

```
python eval.py --run-dir runs_final/<run> --manifests $M/test_*.jsonl \
  --seen test_sdv14 --near test_sdv15 test_wukong \
  --far test_adm test_glide test_vqdm test_midjourney test_biggan \
  --robustness jpeg:95 jpeg:75 jpeg:50 jpeg:96 blur:1 blur:2 rescale:0.5 \
  --aligned-test --save-preds
```

**3. H1 confidence interval (Task 3b):**

```
python scripts/bootstrap_ci.py \
  --vit runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s{0,1,2} \
  --cnn runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s{0,1,2}
```

**4. CLIP probe (Task 6):** extraction takes a few hours on an Apple GPU; the probe takes minutes.

```
python scripts/clip_features.py --manifest-dir $M
python scripts/clip_features.py --manifest-dir $M \
  --conditions aligned jpeg:95 jpeg:75 jpeg:50 jpeg:96 blur:1 blur:2 rescale:0.5
python scripts/clip_probe.py
```

**5. Grad-CAM (Task 7b) and FFT spectra (Task 7a):**

```
python scripts/gradcam.py --manifest-dir $M
python scripts/fft_spectra.py --manifest-dir $M
```

**6. Final tables and Figure 1:**

```
python scripts/aggregate_seeds.py
```

**Other useful commands:**

```
python compare.py runs_final/<run_a> runs_final/<run_b>    # side-by-side per-generator view
python scripts/best_lr.py runs_pace                        # LR selection from the PACE sweep
```
