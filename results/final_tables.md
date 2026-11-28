# Final results

## Table 2: test AUC by generator group (jpeg-crop, mean ± std over seeds)

| Model | Seeds | Seen | Near | Far | Held-out | Gap | Acc | Held-out aligned | Held-out JPEG 95 | JPEG 75 | JPEG 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 3 | 0.998 ± 0.000 | 0.982 ± 0.002 | 0.716 ± 0.008 | 0.792 ± 0.006 | 0.206 ± 0.006 | 0.701 ± 0.002 | 0.897 ± 0.004 | 0.758 ± 0.010 | 0.652 ± 0.008 | 0.633 ± 0.044 |
| ResNet-50 | 3 | 0.999 ± 0.000 | 0.986 ± 0.000 | 0.713 ± 0.044 | 0.791 ± 0.031 | 0.208 ± 0.031 | 0.699 ± 0.013 | 0.898 ± 0.031 | 0.771 ± 0.027 | 0.766 ± 0.027 | 0.750 ± 0.036 |
| DeiT-S | 3 | 0.998 ± 0.000 | 0.989 ± 0.001 | 0.787 ± 0.008 | 0.845 ± 0.006 | 0.154 ± 0.006 | 0.719 ± 0.010 | 0.909 ± 0.002 | 0.812 ± 0.009 | 0.693 ± 0.032 | 0.701 ± 0.030 |
| CLIP ViT-L/14 probe (openai) | 1 | 0.989 | 0.974 | 0.828 | 0.870 | 0.120 | 0.667 | 0.955 | 0.861 | 0.801 | 0.654 |

## Test AUC by generator

| Generator | ResNet-18 | ResNet-50 | DeiT-S | CLIP ViT-L/14 probe (openai) |
|---|---|---|---|---|
| SD1.4 | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.989 |
| SD1.5 | 0.997 ± 0.001 | 0.998 ± 0.001 | 0.998 ± 0.000 | 0.984 |
| Wukong | 0.967 ± 0.005 | 0.974 ± 0.002 | 0.979 ± 0.003 | 0.963 |
| ADM | 0.687 ± 0.012 | 0.697 ± 0.036 | 0.673 ± 0.017 | 0.798 |
| GLIDE | 0.774 ± 0.017 | 0.763 ± 0.063 | 0.859 ± 0.011 | 0.857 |
| VQDM | 0.660 ± 0.008 | 0.694 ± 0.016 | 0.729 ± 0.023 | 0.881 |
| MJ | 0.949 ± 0.002 | 0.953 ± 0.011 | 0.956 ± 0.003 | 0.978 |
| BigGAN | 0.510 ± 0.023 | 0.456 ± 0.095 | 0.721 ± 0.028 | 0.626 |

## Aligned test AUC by generator (fakes re-encoded as JPEG q96)

| Generator | ResNet-18 | ResNet-50 | DeiT-S | CLIP ViT-L/14 probe (openai) |
|---|---|---|---|---|
| SD1.4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |
| SD1.5 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.999 |
| Wukong | 0.996 ± 0.002 | 0.997 ± 0.001 | 0.998 ± 0.000 | 0.997 |
| ADM | 0.866 ± 0.004 | 0.889 ± 0.033 | 0.878 ± 0.005 | 0.970 |
| GLIDE | 0.870 ± 0.015 | 0.878 ± 0.042 | 0.913 ± 0.007 | 0.976 |
| VQDM | 0.868 ± 0.006 | 0.897 ± 0.022 | 0.928 ± 0.005 | 0.989 |
| MJ | 0.969 ± 0.002 | 0.976 ± 0.008 | 0.973 ± 0.001 | 0.994 |
| BigGAN | 0.711 ± 0.011 | 0.650 ± 0.110 | 0.675 ± 0.013 | 0.760 |

## Robustness: mean AUC over all 8 generators

| Model | jpeg:95 | jpeg:75 | jpeg:50 | jpeg:96 | blur:1 | blur:2 | rescale:0.5 |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.788 ± 0.009 | 0.690 ± 0.007 | 0.665 ± 0.041 | 0.793 ± 0.008 | 0.773 ± 0.008 | 0.744 ± 0.004 | 0.784 ± 0.008 |
| ResNet-50 | 0.799 ± 0.023 | 0.792 ± 0.024 | 0.774 ± 0.033 | 0.805 ± 0.023 | 0.786 ± 0.019 | 0.736 ± 0.004 | 0.806 ± 0.023 |
| DeiT-S | 0.835 ± 0.008 | 0.728 ± 0.029 | 0.728 ± 0.029 | 0.844 ± 0.007 | 0.802 ± 0.007 | 0.731 ± 0.014 | 0.793 ± 0.014 |
| CLIP ViT-L/14 probe (openai) | 0.878 | 0.821 | 0.676 | 0.876 | 0.738 | 0.671 | 0.826 |

## Table 3: preprocessing ablation (mean ± std over seeds)

q=96: real and fake JPEG-compressed at test time. Aligned: only non-JPEG fakes re-encoded at q96, so both classes are single-compressed JPEGs.

| Model | Prep | Seeds | Held-out AUC clean | Held-out AUC aligned | AUC all-8 clean | AUC all-8 q=96 | AUC all-8 aligned | Fake acc % clean | Fake acc % q=96 | Fake acc % aligned |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | raw-resize | 1 | 0.707 | 0.722 | 0.744 | 0.734 | 0.756 | 41.9 | 42.5 | 42.5 |
| ResNet-50 | raw-crop | 1 | 0.855 | 0.867 | 0.873 | 0.827 | 0.883 | 52.9 | 50.9 | 50.9 |
| ResNet-50 | jpeg-resize | 1 | 0.704 | 0.749 | 0.741 | 0.741 | 0.780 | 41.3 | 43.3 | 43.3 |
| ResNet-50 | jpeg-crop | 3 | 0.791 ± 0.031 | 0.898 ± 0.031 | 0.817 ± 0.027 | 0.805 ± 0.023 | 0.911 ± 0.027 | 40.5 ± 2.8 | 49.7 ± 3.3 | 49.7 ± 3.3 |

Runs:

- `A_resnet18_in1k_jpeg-crop_lr3e-4_s0`
- `A_resnet18_in1k_jpeg-crop_lr3e-4_s1`
- `A_resnet18_in1k_jpeg-crop_lr3e-4_s2`
- `A_resnet50_in1k_raw-resize_lr1e-4_s0`
- `A_resnet50_in1k_raw-crop_lr1e-4_s0`
- `A_resnet50_in1k_jpeg-resize_lr1e-4_s0`
- `A_resnet50_in1k_jpeg-crop_lr1e-4_s0`
- `A_resnet50_in1k_jpeg-crop_lr1e-4_s1`
- `A_resnet50_in1k_jpeg-crop_lr1e-4_s2`
- `A_deit-s_in1k_jpeg-crop_lr3e-5_s0`
- `A_deit-s_in1k_jpeg-crop_lr3e-5_s1`
- `A_deit-s_in1k_jpeg-crop_lr3e-5_s2`
- `A_clip-l14-probe_jpeg-crop_s0`
