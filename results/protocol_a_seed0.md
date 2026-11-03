# Protocol A results (seed 0, ImageNet-1k init, jpeg-crop)

Setup: train on SD v1.4 (6k fake + 6k real), 10 epochs, batch 64, AdamW (wd 0.01), 5% warmup + cosine.
Same LR grid for every model: {3e-4, 1e-4, 3e-5}. Best LR chosen on seen-generator validation AUC only.
Test: 1,000 fake + 1,000 real per generator. Trained and evaluated on an Apple M5 (fp32).

## Best learning rates
| Model | Best LR | Run |
|---|---|---|
| ResNet-50 | 3e-4 | `A_resnet50_in1k_jpeg-crop_lr3e-4_s0` |
| ResNet-18 | 3e-4 | `A_resnet18_in1k_jpeg-crop_lr3e-4_s0` |
| DeiT-S | 3e-5 | `A_deit-s_in1k_jpeg-crop_lr3e-5_s0` |

## Test AUC by group
| | ResNet-50 | DeiT-S | ResNet-18 |
|---|---|---|---|
| Seen (SD v1.4) | 0.9989 | 0.9985 | 0.9983 |
| Near (SD v1.5, Wukong) | 0.9855 | 0.9896 | 0.9838 |
| Far (ADM, GLIDE, VQDM, Midjourney, BigGAN) | 0.7126 | 0.7934 | 0.7094 |
| All held-out | 0.7906 | 0.8494 | 0.7878 |
| Seen minus held-out | 0.2083 | 0.1490 | 0.2105 |
| Mean accuracy (threshold 0.5) | 0.7037 | 0.7304 | 0.7019 |

## Test AUC by generator
| Generator | ResNet-50 | DeiT-S | ResNet-18 |
|---|---|---|---|
| test_sdv14 (seen) | 0.9989 | 0.9985 | 0.9983 |
| test_sdv15 | 0.9986 | 0.9980 | 0.9974 |
| test_wukong | 0.9723 | 0.9812 | 0.9703 |
| test_midjourney | 0.9490 | 0.9560 | 0.9494 |
| test_glide | 0.7576 | 0.8714 | 0.7848 |
| test_vqdm | 0.7170 | 0.7502 | 0.6506 |
| test_adm | 0.6761 | 0.6896 | 0.6784 |
| test_biggan | 0.4632 | 0.6996 | 0.4840 |

## Robustness (mean AUC over all 8 generators)
| Perturbation | ResNet-50 | DeiT-S | ResNet-18 |
|---|---|---|---|
| JPEG q95 | 0.7765 | 0.8352 | 0.7791 |
| JPEG q75 | 0.7370 | 0.7455 | 0.6965 |
| JPEG q50 | 0.6855 | 0.7514 | 0.6536 |
| Blur sigma 1 | 0.8103 | 0.8098 | 0.7661 |
| Blur sigma 2 | 0.7153 | 0.7453 | 0.7490 |
| Downscale 0.5x | 0.8158 | 0.8087 | 0.7775 |

## Early read
- H1 (DeiT-S better on far generators): supported. +8.1 AUC points over ResNet-50 on far; about +0.4 on near.
- H2 (BigGAN hardest): holds for ResNet-50 and ResNet-18. For DeiT-S, ADM is lower (0.690 vs 0.700).
- Both ResNets score BigGAN below 0.5.

## Caveats
- One seed, no confidence intervals. H1 needs 3 seeds and a bootstrap CI before it counts.
- ResNets' best LR is the top of the grid, DeiT-S's is the bottom, so better values may lie outside it.

## Reproduce
```bash
python eval.py --run-dir runs/<run> --manifests ../genimage_subset/manifests/test_*.jsonl \
  --seen test_sdv14 --near test_sdv15 test_wukong \
  --far test_adm test_glide test_vqdm test_midjourney test_biggan \
  --robustness jpeg:95 jpeg:75 jpeg:50 blur:1 blur:2 rescale:0.5
python compare.py runs/<resnet50_run> runs/<deit-s_run> runs/<resnet18_run>
```
Every number above comes from the runs' `metrics.json` files.