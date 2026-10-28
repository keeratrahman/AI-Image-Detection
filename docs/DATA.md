# Getting GenImage and building our subset

Result: one portable folder (~45 GB) that every teammate and the GPU machine use:

```
genimage_subset/
  images/<generator>/<train|val>/<ai|nature>/...   sampled files, byte-identical to GenImage
  manifests/A_train.jsonl  A_val.jsonl  B_train.jsonl  B_val.jsonl  test_<generator>.jsonl
  manifests/subset_info.json                         the exact arguments used (reproducibility)
```

Configs point at it with `make_configs.py --manifest-dir /path/to/genimage_subset/manifests`.

## Where to download from

| Source                                                                                                  | Status (checked 2026-10-28)                                                                                  |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Official Google Drive ([GitHub](https://github.com/GenImage-Dataset/GenImage))                          | 208 split-zip parts of 3 GiB (~600 GB). Anonymous downloads fail with _"Quota exceeded / Too many users"_.   |
| Official Baidu (code `ztf1`)                                                                            | Needs a Baidu account; very slow from the US.                                                                |
| **Kaggle mirror by vtphatt2 ([GitHub](https://github.com/vtphatt2/GenImage-mirror))** ← **we use this** | One dataset per generator plus a validation dataset. Reliable; needs a free Kaggle account. CC BY-NC-SA 4.0. |

Kaggle datasets we need (Midjourney only appears in tests, so its 227 GB of train data is skipped):

| Kaggle dataset                            | Size    | Used for                       |
| ----------------------------------------- | ------- | ------------------------------ |
| `vtphatt2/genimage-validation`            | 25.5 GB | test sets for all 8 generators |
| `vtphatt2/genimage-stable-diffusion-v1-4` | 96.6 GB | train (protocols A, B)         |
| `vtphatt2/genimage-wukong`                | 97.4 GB | train (B)                      |
| `vtphatt2/genimage-glide`                 | 32.4 GB | train (B)                      |
| `vtphatt2/genimage-vqdm`                  | 38.3 GB | train (B)                      |

Download each one as a single zip, extract only our random sample from it with
`scripts/sample_zip.py`, then delete the zip. Never unzip a whole dataset onto the
"Extreme SSD": it is exFAT with a 1 MB allocation block, so each file takes at least 1 MB
(SD v1.4 alone would allocate ~330 GB).

## Step 1: Environment (once per machine)

```bash
# venv lives outside the SSD: exFAT cannot hold the venv's symlinks
uv venv ~/.venvs/aidet --python 3.11        # or: python3.11 -m venv ~/.venvs/aidet
source ~/.venvs/aidet/bin/activate
uv pip install -r requirements.txt kaggle
python train.py --config configs/smoke/A_resnet18_scratch_jpeg-crop_lr1e-3_s0.yaml
```

## Step 2: Kaggle token (once)

1. Sign in at <https://www.kaggle.com> and open _Settings → API → Create New Token_.
   This downloads `kaggle.json`.
2. `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json`
3. Test it: `kaggle datasets files vtphatt2/genimage-validation | head`

## Step 3: Test sets (validation dataset)

<!-- This is my directory to download the genimage, you can adjust it to match your -->

```bash
DL="/Volumes/Extreme SSD/genimage_dl";  POOL="/Volumes/Extreme SSD/genimage_pool"
kaggle datasets download vtphatt2/genimage-validation -p "$DL"
python scripts/sample_zip.py --zip "$DL/genimage-validation.zip" --out "$POOL" --splits val --list-only
python scripts/sample_zip.py --zip "$DL/genimage-validation.zip" --out "$POOL" --splits val
rm "$DL/genimage-validation.zip"
```

`--list-only` should show `('<gen>', 'val', 'ai')` and `('<gen>', 'val', 'nature')` for all 8
generators. If a generator is missing, stop and tell the team.

## Step 4: Training generators (one at a time)

```bash
for D in genimage-stable-diffusion-v1-4 genimage-glide genimage-vqdm genimage-wukong; do
  kaggle datasets download vtphatt2/$D -p "$DL" &&
  python scripts/sample_zip.py --zip "$DL/$D.zip" --out "$POOL" --splits train &&
  rm "$DL/$D.zip"
done
```

`--splits train` matters: test images always come from the validation dataset (step 3), so
they cannot leak into training. Each training generator contributes 7,000 fake + 7,000 real
images (val + train). Re-running is safe; files that already exist are skipped.

## Step 5: Write the subset and manifests

```bash
SUBSET="/Volumes/Extreme SSD/genimage_subset"
python scripts/make_subset.py --src "$POOL" --dest "$SUBSET" --dry-run   # check counts
python scripts/make_subset.py --src "$POOL" --dest "$SUBSET"
```

Expected output: `test_<gen>.jsonl` with 2,000 images for each of the 8 generators,
`A_train` 12,000, `A_val` 2,000, `B_train` 12,000 and `B_val` 8,000. After that the pool
folder can be deleted.

## Step 6: Check it on real data

```bash
python make_configs.py --models resnet18 --init in1k --lrs 1e-4 --preps jpeg-crop --epochs 1 \
    --batch-size 16 --num-workers 2 --wandb-mode disabled \
    --manifest-dir "$SUBSET/manifests" --output-dir runs_smoke --out configs/smoke
python train.py --config configs/smoke/A_resnet18_in1k_jpeg-crop_lr1e-4_s0.yaml
```

## Step 7: Share it

```bash
tar -cf genimage_subset.tar -C "/Volumes/Extreme SSD" genimage_subset
rsync -ah --progress genimage_subset.tar <user>@<gpu-host>:/path/to/scratch/   # then: tar -xf genimage_subset.tar
```

Manifest paths are relative to the subset folder, so they work wherever it is unpacked.
Teammates can also rebuild the same subset themselves: the same seed and sizes give the same files.
