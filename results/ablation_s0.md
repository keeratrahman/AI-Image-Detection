# Final results

## Table 2: test AUC by generator group (jpeg-crop, mean ± std over seeds)

| Model | Seeds | Seen | Near | Far | Held-out | Gap | Acc | Held-out aligned | Held-out JPEG 95 | JPEG 75 | JPEG 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 1 | 0.999 | 0.986 | 0.717 | 0.794 | 0.205 | 0.706 | 0.899 | 0.762 | 0.755 | 0.731 |
| DeiT-S | 1 | 0.998 | 0.990 | 0.793 | 0.849 | 0.149 | 0.730 | 0.911 | 0.812 | 0.713 | 0.726 |

## Test AUC by generator

| Generator | ResNet-50 | DeiT-S |
|---|---|---|
| SD1.4 | 0.999 | 0.998 |
| SD1.5 | 0.998 | 0.998 |
| Wukong | 0.974 | 0.981 |
| ADM | 0.689 | 0.690 |
| GLIDE | 0.768 | 0.871 |
| VQDM | 0.701 | 0.750 |
| MJ | 0.953 | 0.956 |
| BigGAN | 0.471 | 0.700 |

## Aligned test AUC by generator (fakes re-encoded as JPEG q96)

| Generator | ResNet-50 | DeiT-S |
|---|---|---|
| SD1.4 | 1.000 | 1.000 |
| SD1.5 | 0.999 | 0.999 |
| Wukong | 0.996 | 0.998 |
| ADM | 0.891 | 0.884 |
| GLIDE | 0.883 | 0.921 |
| VQDM | 0.906 | 0.932 |
| MJ | 0.977 | 0.972 |
| BigGAN | 0.638 | 0.670 |

## Robustness: mean AUC over all 8 generators

| Model | jpeg:95 | jpeg:75 | jpeg:50 | jpeg:96 | blur:1 | blur:2 | rescale:0.5 |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.792 | 0.782 | 0.756 | 0.798 | 0.783 | 0.732 | 0.802 |
| DeiT-S | 0.835 | 0.746 | 0.751 | 0.845 | 0.810 | 0.745 | 0.809 |

## Table 3: preprocessing ablation (mean ± std over seeds)

q=96: real and fake JPEG-compressed at test time. Aligned: only non-JPEG fakes re-encoded at q96, so both classes are single-compressed JPEGs.

| Model | Prep | Seeds | Held-out AUC clean | Held-out AUC aligned | AUC all-8 clean | AUC all-8 q=96 | AUC all-8 aligned | Fake acc % clean | Fake acc % q=96 | Fake acc % aligned |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | raw-resize | 1 | 0.707 | 0.722 | 0.744 | 0.734 | 0.756 | 41.9 | 42.5 | 42.5 |
| ResNet-50 | raw-crop | 1 | 0.855 | 0.867 | 0.873 | 0.827 | 0.883 | 52.9 | 50.9 | 50.9 |
| ResNet-50 | jpeg-resize | 1 | 0.704 | 0.749 | 0.741 | 0.741 | 0.780 | 41.3 | 43.3 | 43.3 |
| ResNet-50 | jpeg-crop | 1 | 0.794 | 0.899 | 0.819 | 0.798 | 0.911 | 42.1 | 50.8 | 50.8 |
| DeiT-S | raw-resize | 1 | 0.756 | 0.744 | 0.786 | 0.770 | 0.776 | 41.1 | 40.6 | 40.6 |
| DeiT-S | raw-crop | 1 | 0.898 | 0.859 | 0.910 | 0.824 | 0.876 | 54.3 | 52.0 | 52.0 |
| DeiT-S | jpeg-resize | 1 | 0.741 | 0.758 | 0.773 | 0.769 | 0.788 | 42.0 | 43.7 | 43.7 |
| DeiT-S | jpeg-crop | 1 | 0.849 | 0.911 | 0.868 | 0.845 | 0.922 | 47.7 | 59.9 | 59.9 |

Runs:

- `A_resnet50_in1k_raw-resize_lr1e-4_s0`
- `A_resnet50_in1k_raw-crop_lr1e-4_s0`
- `A_resnet50_in1k_jpeg-resize_lr1e-4_s0`
- `A_resnet50_in1k_jpeg-crop_lr1e-4_s0`
- `A_deit-s_in1k_raw-resize_lr3e-5_s0`
- `A_deit-s_in1k_raw-crop_lr3e-5_s0`
- `A_deit-s_in1k_jpeg-resize_lr3e-5_s0`
- `A_deit-s_in1k_jpeg-crop_lr3e-5_s0`
