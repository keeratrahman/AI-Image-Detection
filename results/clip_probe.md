# Task 6: CLIP ViT-L/14 linear probe (Ojha et al., 2023 reference)

Frozen CLIP image features plus a logistic-regression head, trained and tested on the same Protocol A
splits as our end-to-end models. All numbers come from
`runs_final/A_clip-l14-probe_jpeg-crop_s0/metrics.json`.

## Setup

- **Features:** open_clip `ViT-L-14`, `pretrained="openai"`, frozen. 768-d image embedding,
  L2-normalized before the probe.
- **Preprocessing:** our `jpeg-crop` eval view (upsample if the short side is under 224, then a 224 center
  crop, no flip) with CLIP's mean/std instead of ImageNet's.
  - Training features (`A_train`, 12k) and validation features (`A_val`, 2k) re-encode non-JPEG fakes at
    JPEG q96, as in `jpeg-crop` training.
  - The 8 test sets are left untouched, as in `eval.py`.
  - Center crop is also used for the training features, so features are extracted only once.
- **Probe:** sklearn `LogisticRegression` (lbfgs, L2), fit on `A_train` only.
  - C is chosen from {0.01, 0.1, 1, 10} by `A_val` AUC.
  - Ties within 0.0005 are broken by lowest `A_val` log loss, the same rule as the LR sweep.
- **Evaluation:** the same `eval.score` / `eval.summarize` code and the same generator groups as every
  other run. Per-image predictions are in `preds/<condition>/` (not in git).
- **Seeds:** one run. The probe is deterministic given the features, so seeds 1–2 would give the same
  result.

| C | A_val AUC | A_val log loss |
|---|---|---|
| 0.01 | 0.98508 | 0.492 |
| 0.1 | 0.99640 | 0.223 |
| 1 | 0.99909 | 0.080 |
| **10 (chosen)** | **0.99957** | **0.032** |

C = 1 and C = 10 tie on AUC (within 0.0005), and C = 10 wins on loss. **C = 10 is the edge of the
grid**, so report it as a limitation, as with the DeiT-S and ResNet-18 learning rates.

## Results (clean test)

Our models: mean ± std over 3 seeds. CLIP: 1 run.

| Model | Seen | Near | Far | Held-out | Seen − held-out gap | Acc |
|---|---|---|---|---|---|---|
| ResNet-18 | 0.998 | 0.982 | 0.716 ± 0.008 | 0.792 ± 0.006 | 0.206 | 0.701 |
| ResNet-50 | 0.999 | 0.986 | 0.713 ± 0.044 | 0.791 ± 0.031 | 0.208 | 0.699 |
| DeiT-S | 0.998 | 0.989 | 0.787 ± 0.008 | 0.845 ± 0.006 | 0.154 | 0.719 |
| **CLIP probe** | 0.989 | 0.974 | **0.828** | **0.870** | **0.120** | 0.667 |

Per generator (AUC):

| | SD1.4 | SD1.5 | Wukong | ADM | GLIDE | VQDM | MJ | BigGAN |
|---|---|---|---|---|---|---|---|---|
| DeiT-S | 0.998 | 0.998 | 0.979 | 0.673 | 0.859 | 0.729 | 0.956 | **0.721** |
| ResNet-50 | 0.999 | 0.998 | 0.974 | 0.697 | 0.763 | 0.694 | 0.953 | 0.456 |
| CLIP probe | 0.989 | 0.984 | 0.963 | **0.798** | 0.857 | **0.881** | **0.978** | 0.626 |

Extra test conditions (mean AUC over all 8 generators):

| Model | Clean | Aligned | JPEG 95 | JPEG 75 | JPEG 50 | Blur σ=1 | Blur σ=2 | Rescale 0.5 |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.817 | 0.911 | 0.799 | 0.792 | 0.774 | 0.786 | 0.736 | 0.806 |
| DeiT-S | 0.864 | – | 0.835 | 0.728 | 0.728 | 0.802 | 0.731 | 0.793 |
| CLIP probe | 0.885 | 0.961 | 0.878 | 0.821 | 0.676 | 0.738 | 0.671 | 0.826 |

DeiT-S has no aligned-test result in `runs_final` yet, so its cell is empty.

## Takeaways

1. **CLIP generalizes best to unseen generators.** It has the highest held-out AUC (0.870 vs. 0.845 for
   DeiT-S and 0.791 for ResNet-50) and the smallest seen-versus-held-out gap (0.120). Most of the gain
   comes from ADM (+0.10 over the best trained model) and VQDM (+0.15). This matches Ojha et al.: a
   frozen, general-purpose feature space does not overfit to SD v1.4's fingerprint, because the probe
   cannot change the features.
2. **The ranking CLIP (ViT-L/14) > DeiT-S > ResNet-50 on far generators matches H1's direction.**
   Transformer features transfer better than CNN features. But CLIP differs from DeiT-S in size,
   pretraining data and training method, so it is a reference point, not a controlled test of
   architecture.
3. **CLIP pays for this on the seen and near generators.** It is about 0.01–0.015 AUC lower there, and
   its accuracy at threshold 0 is the lowest of all models (0.667). Its scores rank images well, but the
   threshold learned on SD v1.4 calls many unseen fakes real.
4. **CLIP is not uniformly better.** DeiT-S still wins on BigGAN (0.721 vs. 0.626), the only GAN in the
   test set. So the BigGAN gain that drives the H1 result is specific to the trained DeiT-S, not just
   "a transformer".
5. **CLIP is the least robust to strong degradations.** It leads at mild JPEG (q95, q75) and at rescale,
   but drops to the lowest score at JPEG q50 (0.676) and under blur (0.738 / 0.671). Its features seem
   to rely partly on fine detail that strong compression and blur remove.
6. **Part of CLIP's clean score is still the format cue.** Its AUC rises from 0.885 to 0.961 on the
   aligned test, where non-JPEG fakes are re-encoded at q96 so both classes are single-compressed JPEGs.
   Clean test sets have JPEG reals and mostly PNG fakes, while the probe was trained on aligned fakes,
   so the clean test is a format shift for it. ResNet-50 shows the same pattern (0.817 → 0.911).

## Limitations

- **C at grid edge:** C = 10 is the largest value tried.
- **Single run:** there is no seed variance, and the probe is not fit on Protocol B.
- **Comparison caveat:** compare CLIP against our models as a reference row, not as a matched
  architecture comparison.

## Reproduce

```
python scripts/clip_features.py --manifest-dir ../genimage_subset/manifests
python scripts/clip_features.py --manifest-dir ../genimage_subset/manifests \
  --conditions aligned jpeg:95 jpeg:75 jpeg:50 jpeg:96 blur:1 blur:2 rescale:0.5
python scripts/clip_probe.py
python scripts/aggregate_seeds.py
```
