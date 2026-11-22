# H1: DeiT-S minus ResNet-50 AUC gap (far generators, clean test)

**gap = 7.5 [6.7, 8.2]  seed std = 3.8** (AUC points; 95% paired bootstrap CI, 10000 resamples)

H1 rule: gap >= 3 points and CI excludes 0 -> **SUPPORTED**

| generator | DeiT-S AUC | ResNet-50 AUC | gap (pts) |
|---|---|---|---|
| test_adm | 0.6728 | 0.6970 | -2.4 |
| test_glide | 0.8585 | 0.7634 | +9.5 |
| test_vqdm | 0.7287 | 0.6942 | +3.4 |
| test_midjourney | 0.9557 | 0.9529 | +0.3 |
| test_biggan | 0.7208 | 0.4559 | +26.5 |
| **mean** | 0.7873 | 0.7127 | +7.5 |

Seeds: 3 DeiT-S, 3 ResNet-50.
Per-seed gaps: +7.7, +3.6, +11.1

Runs:
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0`
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s1`
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s2`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s0`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s1`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s2`

Command: `python scripts/bootstrap_ci.py --vit runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0 runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s1 runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s2 --cnn runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s0 runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s1 runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s2`
Generated: 2026-11-22T16:21:46+00:00
