# H4 ablation: JPEG alignment x crop/resize (Protocol A, seed 0)

Setup: ImageNet-1k init, trained on SD v1.4, 10 epochs, batch 64. Each model used its best LR from the
sweep (ResNet-50 3e-4, DeiT-S 3e-5), not retuned per cell. Run names follow
`A_<model>_in1k_<prep>_lr<lr>_s0`, with prep in {raw-resize, raw-crop, jpeg-resize, jpeg-crop}.
Test images are untouched in the clean test. "JPEG q96" recompresses every test image (real and fake).

## Held-out AUC (7 held-out generators)
| | ResNet-50 resize | ResNet-50 crop | DeiT-S resize | DeiT-S crop |
|---|---|---|---|---|
| Raw | 0.7149 | 0.8751 | 0.7560 | 0.8976 |
| JPEG-aligned | 0.7195 | 0.7906 | 0.7415 | 0.8494 |

## All-8 AUC, clean vs. JPEG q96 on every test image
| Model | Prep | Clean | JPEG q96 | Change |
|---|---|---|---|---|
| ResNet-50 | raw-resize | 0.7502 | 0.7432 | -0.0070 |
| ResNet-50 | raw-crop | 0.8907 | 0.8346 | -0.0561 |
| ResNet-50 | jpeg-resize | 0.7543 | 0.7485 | -0.0058 |
| ResNet-50 | jpeg-crop | 0.8166 | 0.7857 | -0.0309 |
| DeiT-S | raw-resize | 0.7860 | 0.7696 | -0.0164 |
| DeiT-S | raw-crop | 0.9104 | 0.8241 | -0.0863 |
| DeiT-S | jpeg-resize | 0.7734 | 0.7694 | -0.0040 |
| DeiT-S | jpeg-crop | 0.8681 | 0.8450 | -0.0231 |

## Mean fake accuracy (threshold 0.5), clean vs. JPEG q96
| Model | Prep | Clean | JPEG q96 |
|---|---|---|---|
| ResNet-50 | raw-resize | 0.411 | 0.416 |
| ResNet-50 | raw-crop | 0.499 | 0.436 |
| ResNet-50 | jpeg-resize | 0.403 | 0.420 |
| ResNet-50 | jpeg-crop | 0.415 | 0.509 |
| DeiT-S | raw-resize | 0.411 | 0.406 |
| DeiT-S | raw-crop | 0.543 | 0.520 |
| DeiT-S | jpeg-resize | 0.420 | 0.437 |
| DeiT-S | jpeg-crop | 0.477 | 0.599 |

## Findings
- **H4(a) supported.** Crop beats resize by 7 to 16 held-out AUC points for both models, and also after JPEG q96.
- **JPEG alignment alone does not help.** jpeg-resize is about equal to raw-resize.
- **Shortcut evidence.** raw-crop has the highest clean AUC but the largest drop under JPEG q96
  (-0.056, -0.086) vs. jpeg-crop (-0.031, -0.023). After compression DeiT-S jpeg-crop (0.845) beats
  raw-crop (0.824); for ResNet-50, raw-crop (0.835) still leads jpeg-crop (0.786).
- **H4(b) not supported at the plan's magnitude.** No raw cell loses 20 fake-accuracy points (the largest drop is 6).
  Fake accuracy is a weak metric here: it is only 0.40 to 0.54 before compression because the 0.5
  threshold is miscalibrated on unseen generators.

## Caveats
- One seed and no confidence intervals; differences of about 1 point are noise.
- LR was tuned on jpeg-crop only, which may favor that cell.
- JPEG q96 also recompresses real images, so it only approximates a debiased test set.
- The clean test is mismatched for the aligned models: they train on JPEG-re-encoded fakes but are tested
  on untouched PNG fakes. This probably explains why their fake accuracy rises under JPEG q96.

## Reproduce
```bash
python eval.py --run-dir runs/<run> --manifests ../genimage_subset/manifests/test_*.jsonl \
  --seen test_sdv14 --near test_sdv15 test_wukong \
  --far test_adm test_glide test_vqdm test_midjourney test_biggan --robustness jpeg:96
python ablation.py
```
Every number above comes from the runs' `metrics.json` files.