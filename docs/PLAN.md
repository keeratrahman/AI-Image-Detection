# Project plan: Does architecture affect cross-generator generalization?

Draft by Nhi, 2026-10-28. **Dates and owner assignments are proposals; confirm as a team.**
The mid plan (Part 2) is the immediate goal. Part 3 extends it to the final report.

---

## 0. TA feedback → what we change

| TA comment | Fix (section) |
|---|---|
| No testable prediction (−0.5) | Hypotheses H1–H5 with a direction, a metric and a reason (§1.3) |
| Inputs/outputs not specified (−0.5) | Exact tensor shape, resolution, crop-vs-resize, normalization (§1.4) |
| Motivation is brief (−0.5) | Stakes paragraph: misinformation, provenance, moderation (§1.1) |
| Exact architectures + init | Parameter-matched ResNet-50 vs DeiT-S, both ImageNet-1k init; scratch as a separate factor (§1.4) |
| Resizing destroys high-frequency artifacts | Native-resolution random crop is the default; resize is tested as a factor (§1.4, H4) |
| Pretrained vs. from-scratch confound | Every architecture comparison uses matched init (§1.4, H3) |
| What does it add beyond GenImage's backbone table? | Controlled comparison + bias fixes + multi-generator protocol (§1.2) |
| Prior-method reference point | CLIP ViT-L/14 linear probe (Ojha et al., 2023) on the same splits (§1.5) |
| Fix the held-out generators | Protocols A and B are fixed in `scripts/make_subset.py` (§1.4) |
| Add mitigations for the risks | Risk table (§4) |

---

## 1. Revised method

### 1.1 Motivation (new paragraph for the report)
Photorealistic generators are free and fast. Synthetic images are already used in political
misinformation, fake news photos, fraud (fake IDs, insurance claims) and harassment.
Provenance standards such as C2PA watermarks only help when the generator cooperates, and the
metadata is stripped by a screenshot. Platforms and fact-checkers therefore need *passive*
detectors. The open problem is that a detector meets images from generators that did not exist
when it was trained. A detector that only recognizes its training generator gives false
confidence, which is worse than no detector. We study which design choice, architecture or
preprocessing, makes detectors fail less on unseen generators.

### 1.2 What we add beyond GenImage's own backbone table
GenImage (Zhu et al., 2023) reports ResNet-50, DeiT-S and Swin-T trained on SD v1.4. Those
backbones differ in more than architecture, and the protocol has two known biases:
1. **Controlled architecture comparison.** Our CNN and ViT have about the same number of
   parameters (ResNet-50 25.6M vs. DeiT-S 22M), the same ImageNet-1k supervised init, the same
   data, schedule, learning-rate budget and seeds.
2. **Bias controls.** GenImage's real images are JPEG (ImageNet) and most fakes are PNG, and
   image sizes differ by class. Detectors can learn "is it JPEG-compressed?" instead of
   generation artifacts (Grommelt et al., 2024, *Fake or JPEG?*). We test JPEG alignment and
   crop-vs-resize as explicit factors (H4).
3. **Multi-generator training (Protocol B)** with total training size held equal to
   Protocol A. This tests whether generator *diversity* helps one architecture more.
4. **Robustness and a frozen-feature reference.** Results under JPEG, blur and downscaling,
   plus a CLIP linear probe, show where end-to-end training stands relative to Ojha et al.

### 1.3 Hypotheses (testable)
"Held-out AUC" means ROC-AUC on the test split of a generator not used in training. *Near*
generators share SD v1.4's latent-diffusion decoder (SD v1.5, Wukong). *Far* generators are
ADM, GLIDE, VQDM, Midjourney and BigGAN.

- **H1 (architecture).** Under Protocol A with matched init, DeiT-S has higher mean AUC on the
  far generators than ResNet-50, by at least 3 AUC points averaged over 3 seeds. On near
  generators the gap is about 0. *Reason:* convolutions have a strong locality bias and readily
  fit the periodic high-frequency fingerprint of SD's VAE upsampler, which is specific to that
  generator. ViT's global attention and large patch embedding weight these per-pixel patterns
  less and rely more on image-level cues, which transfer better. GenImage's table points the same
  way, but without matched controls. *Rejected if* the 95% CI of the gap includes 0.
- **H2 (GAN is hardest).** For every model, BigGAN has the lowest held-out AUC under
  Protocol A. *Reason:* GAN upsampling artifacts differ from diffusion artifacts, and BigGAN's
  128×128 images must be upsampled to 224, which changes their frequency content.
- **H3 (pretraining confound).** Training from scratch reduces held-out AUC more than seen-generator
  AUC, and the drop is larger for DeiT-S than for ResNet-50. This would show why a pretrained
  CNN vs. scratch ViT comparison is invalid. *Reason:* ViTs lack locality priors and need far
  more data than our subset.
- **H4 (preprocessing 2×2: JPEG alignment × crop/resize).** (a) Resize lowers held-out AUC
  vs. crop for both architectures. (b) Without JPEG alignment, accuracy on fakes drops by more
  than 20 points when test fakes are JPEG-compressed at q=96; with alignment it drops by less
  than 5. *Reason:* resizing smooths away high-frequency artifacts, and unaligned training
  teaches the compression shortcut.
- **H5 (diversity, Protocol B).** Training on 4 generators instead of 1 at equal data size
  improves mean held-out AUC for both models. The gain is larger for ResNet-50, which narrows
  the H1 gap. *Reason:* diversity prevents fitting a single generator's fingerprint, which is
  the CNN's main failure mode under H1.

### 1.4 Fixed experimental setup

**Network input/output.** Input is an RGB float32 tensor (B, 3, 224, 224), normalized with
ImageNet mean/std. Output is one logit per image; σ(logit) = P(fake). Loss is
`BCEWithLogitsLoss`, with label 1 = fake.

**Preprocessing (`aidet/genimage.py`).**
- *Default `jpeg-crop`:* fakes that are not already JPEG are re-encoded in memory as JPEG q=96
  to match ImageNet's typical quality. Training uses a random 224×224 crop at native resolution
  plus horizontal flip; evaluation uses a center crop. Images are never downscaled. Images with a
  short side under 224 (BigGAN, 128 px) are upsampled with bicubic interpolation; this is the
  only resampling, and we report it as a limitation.
- *Factor variants:* `raw-*` skips the JPEG re-encode, and `*-resize` resizes the whole image to
  224×224 (bicubic) instead of cropping.

**Models (`aidet/models.py`).**

| Name | Role | Params | Init |
|---|---|---|---|
| `resnet50` | CNN for H1 (main) | 25.6M | ImageNet-1k (torchvision V1) and scratch |
| `deit-s` (timm `deit_small_patch16_224`) | ViT for H1 (main) | 22M | ImageNet-1k (supervised) and scratch |
| `resnet18` | Cheap baseline from the proposal | 11.7M | ImageNet-1k |
| `cnn-small` | Plain 4-block conv net | ~1M | scratch |
| CLIP ViT-L/14 + linear | Reference (Ojha et al.) | frozen | OpenAI CLIP |

**Training.** AdamW (wd 0.01), 5% warmup + cosine schedule, bf16 on CUDA, 10 epochs, batch 64.
Every model gets the same 3-value learning-rate grid (ResNet {3e-4, 1e-4, 3e-5}, DeiT
{1e-4, 5e-5, 1e-5}, i.e. the same budget). Checkpoints are selected on seen-generator val AUC
only; held-out generators are never used for selection. Each final configuration runs with
3 seeds.

**Protocols and splits (`scripts/make_subset.py`).**

| Protocol | Train generators | Held out (test only) |
|---|---|---|
| **A** (GenImage standard) | SD v1.4 | SD v1.5, Wukong (near); ADM, GLIDE, VQDM, Midjourney (far diffusion); BigGAN (GAN) |
| **B** (diverse) | SD v1.4, Wukong, GLIDE, VQDM | SD v1.5 (near); ADM, Midjourney (far diffusion); BigGAN (GAN) |

- Train: 6,000 fake + 6,000 real per protocol, split evenly across that protocol's generators.
- Val (model selection): 1,000 fake + 1,000 real per training generator, carved out of GenImage
  *train*.
- Test: 1,000 fake + 1,000 real per generator from GenImage's official *val* split. The same
  test files are used across all protocols and models.
- Real images come from the same generator folder as the fakes they are paired with, so the
  ImageNet classes are matched.

**Metrics.** The primary metric is ROC-AUC per generator, reported as the mean over near, far
and all held-out generators. We also report accuracy at threshold 0.5, split into real accuracy
and fake accuracy as GenImage does, plus F1, precision and recall, and the gap between seen and
held-out AUC. Uncertainty is mean ± std over 3 seeds, with a bootstrap 95% CI for the H1 gap.

**Robustness (test-time only, applied to real and fake images).** JPEG q ∈ {95, 75, 50},
Gaussian blur σ ∈ {1, 2}, and 0.5× downscale followed by upsampling back.

### 1.5 Reference point
CLIP ViT-L/14 image embeddings (frozen) with a logistic-regression head, trained on the same
Protocol A/B train manifests. It shows how much a frozen foundation-model feature generalizes
compared with our end-to-end models.

---

## 2. MID PLAN: first goal

**Target result:** the Protocol A table (8 generators × 3 models) with H1 and H2 answered at
1 seed, plus one robustness curve.

| # | Deliverable | Owner (proposed) | Due (proposed) |
|---|---|---|---|
| M1 | Clean env + smoke test passes on every member's machine and on the GPU | everyone | Oct 31 |
| M2 | GenImage downloaded and pooled; manifests for A and B on shared storage | **Nhi** | Nov 4 |
| M3 | `deit-s` and `cnn-small` registered in `aidet/models.py` | Ahmed M | Nov 4 |
| M4 | `eval.py`: per-generator metrics on all `test_*.jsonl` + robustness perturbations | Ahmed A | Nov 7 |
| M5 | LR sweep (3 LRs) for ResNet-50 and DeiT-S on A / `jpeg-crop` / in1k, seed 0 | Keerat | Nov 9 |
| M6 | Best-LR runs: ResNet-50, DeiT-S, ResNet-18 → full test table + JPEG robustness | everyone | Nov 12 |
| M7 | Mid-point write-up: revised method (§1), first table, early read on H1/H2 | everyone | Nov 14 |

**Compute estimate:** 12k training images × 10 epochs × ~10 runs. That is minutes per run on any
modern GPU; image decoding, not the GPU, will be the bottleneck.

**Mid plan is done when:** the Protocol A results table has numbers in every cell, every run's
`metrics.json` and `best.pt` are on shared storage, and anyone on the team can reproduce a run from
its config file name.

## 3. Final plan (after the mid plan)

| Week of | Work |
|---|---|
| Nov 16 | Seeds 1–2 for the mid-plan runs; H4 2×2 for ResNet-50 and DeiT-S (A) |
| Nov 23 | H3 scratch runs; Protocol B (H5); CLIP linear probe (A and B) |
| Nov 30 | Full robustness grid; error analysis (Grad-CAM / attention maps on mistakes, failure cases per generator, FFT spectra of real vs. fake) |
| Dec 7 | Final report and figures |

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset too big to download/share (hundreds of GB, Drive quotas) | Download one generator at a time, pool a fixed seeded subset (`--pool-only`), then delete the archive. Share the subset (~45 GB) and its manifests, not the raw download. Fallback: Baidu mirror, or pull from a teammate's copy. |
| Shortcut learning (JPEG / image size) inflates results | JPEG alignment + native-resolution crops by default; H4 measures the effect directly |
| Pretraining confound | All H1 comparisons use matched init; scratch is its own factor (H3) |
| Unfair hyperparameter tuning | Same LR budget and schedule for every model; select on seen-generator val only |
| Single-seed noise | 3 seeds for final configurations; report CIs |
| Models fail on unseen generators | That is a valid finding. Report the seen vs. held-out gap and the CLIP reference so a low number is still interpretable. |
| Sharing models/results | Configs in git; checkpoints and `metrics.json` in one shared folder (`output_dir`); runs logged to a shared W&B project |
| Future generators (2024+) not in GenImage | Stated as a limitation. Optional stretch: 500 images from a recent open model (e.g. SDXL/FLUX) as an extra test set. |
| GPU access lost or queue delays near deadline | Mid-plan runs are small; keep a Colab fallback; finish the LR sweep early |

## 5. References to add
- Grommelt, P. et al. *Fake or JPEG? Revealing Common Biases in Generated Image Detection
  Datasets.* ECCV Workshops 2024 (arXiv:2403.17608). Their cleaned-GenImage code: unbiased-genimage.org.
- Touvron, H. et al. *Training data-efficient image transformers (DeiT).* ICML 2021.
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
