# Task 7b: Grad-CAM maps for ResNet-50 and DeiT-S

Figure: `results/figures/gradcam.png` (and `.pdf`). The chosen images and their logits are in
`results/figures/gradcam_examples.json`.

## Method

- **Models:** seed-0 checkpoints of the final runs:
  - `runs_final/A_resnet50_in1k_jpeg-crop_lr1e-4_s0`
  - `runs_final/A_deit-s_in1k_jpeg-crop_lr3e-5_s0`
- **Target:** the single "fake" logit (`ClassifierOutputTarget(0)`). Warm regions are evidence for
  "fake". An all-blue map means the layer found no evidence for "fake" anywhere.
- **ResNet-50:** Grad-CAM at `layer4[-1]` (7×7 grid).
- **DeiT-S:** Grad-CAM at `blocks[-1].norm1`. The 196 patch tokens are reshaped to 14×14, and the CLS
  token is dropped.
- **Input:** each image goes through the same eval view as `eval.py` (224 center crop, ImageNet
  normalization). The script re-predicts every image and checks the result against the saved
  per-image logit. All 12 matched.
- **Image selection:** done automatically from the saved clean predictions (`preds/clean/`), fake
  images only. There are four images per category (12 rows):

| Rows | Generator | Category | Candidates | Ranked by | ResNet-50 p(fake) | DeiT-S p(fake) |
|---|---|---|---|---|---|---|
| 1–4 | SD v1.4 (seen) | both right | 970 of 1,000 | most confident for both | 1.00 ×4 | 1.00 ×4 |
| 5–8 | GLIDE | DeiT-S right, ResNet-50 wrong | 149 of 1,000 | largest DeiT-S minus ResNet-50 logit | 0.00 ×4 | 0.96, 0.99, 0.89, 0.95 |
| 9–12 | BigGAN | both wrong | 985 of 1,000 | most confidently wrong | 0.00 ×4 | 0.00 ×4 |

"Candidates" is how many of the 1,000 fakes in that test set fit the category.

## Takeaway

ResNet-50's evidence for "fake" sits in a few large blobs, often on the background or image border
rather than the object. DeiT-S's evidence is spread over many patches across the whole image. On all
four GLIDE images that only DeiT-S gets right, ResNet-50's map is almost empty (at most a small corner
spot), while DeiT-S finds scattered evidence across the image, including on a near-blank page of
text. This fits H1: the CNN learned a
localized cue that does not appear in GLIDE images, and the ViT relies on cues spread across the
image, which transfer better. On BigGAN, where both models are confidently wrong, ResNet-50's map is
empty and DeiT-S's has only a few small spots, so neither model finds much it associates with SD v1.4
fakes. This matches BigGAN being
the hardest generator: AUC 0.46 for ResNet-50 and 0.72 for DeiT-S, averaged over seeds.

## Caveats

- **Few images:** 12 images are illustrations, not statistical evidence. Each category's images were
  picked by an extreme score, so they show the clearest cases.
- **Different map resolution:** ResNet-50's Grad-CAM map is 7×7 and DeiT-S's is 14×14, so ResNet-50
  maps look blobbier partly because of resolution.
- **Grad-CAM on ViTs is noisier than on CNNs:** attention rollout is a common alternative and was not
  run.
- **Per-image scaling:** each map is scaled per image, so compare where the warm regions are, not how
  bright they are across rows.

## Reproduce

```
python scripts/gradcam.py --manifest-dir ../genimage_subset/manifests
```

This needs `best.pt` and `preds/clean/` for both runs. Neither is in git; get them from the shared
storage.
