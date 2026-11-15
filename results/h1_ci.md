# H1: DeiT-S minus ResNet-50 AUC gap (far generators, clean test)

**gap = 6.4 [5.7, 7.2]  seed std = 1.6** (AUC points; 95% paired bootstrap CI, 10000 resamples)

H1 rule: gap >= 3 points and CI excludes 0 -> **SUPPORTED**

| generator | DeiT-S AUC | ResNet-50 AUC | gap (pts) |
|---|---|---|---|
| test_adm | 0.6728 | 0.7177 | -4.5 |
| test_glide | 0.8585 | 0.7776 | +8.1 |
| test_vqdm | 0.7287 | 0.7277 | +0.1 |
| test_midjourney | 0.9557 | 0.9484 | +0.7 |
| test_biggan | 0.7208 | 0.4427 | +27.8 |
| **mean** | 0.7873 | 0.7228 | +6.4 |

Seeds: 3 DeiT-S, 3 ResNet-50.
Per-seed gaps: +8.1, +6.3, +5.0

Runs:
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0`
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s1`
- ViT: `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s2`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s0`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s1`
- CNN: `runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s2`

Command: `python scripts/bootstrap_ci.py --vit runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0 runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s1 runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s2 --cnn runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s0 runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s1 runs_final/A_resnet50_in1k_jpeg-crop_lr3e-4_s2`
Generated: 2026-11-15T01:44:50+00:00
