import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from aidet.naming import prep_flags

IMAGE_SIZE = 224
JPEG_QUALITY = 96  # most common quality factor in ImageNet's real JPEGs
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    root = manifest_path.resolve().parent.parent
    records = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                p = Path(rec["path"])
                rec["path"] = str(p if p.is_absolute() else root / p)
                records.append(rec)
    return records


def jpeg_reencode(img: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class MinShortSide:
    """Upsample only when the short side is below `size`; never downsample."""

    def __init__(self, size: int):
        self.size = size
        self.resize = T.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, img):
        return self.resize(img) if min(img.size) < self.size else img


def build_transform(crop: bool, train: bool):
    if crop:
        spatial = [MinShortSide(IMAGE_SIZE),
                   T.RandomCrop(IMAGE_SIZE) if train else T.CenterCrop(IMAGE_SIZE)]
    else:
        spatial = [T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True)]
    if train:
        spatial.append(T.RandomHorizontalFlip())
    return T.Compose(spatial + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class GenImageDataset(Dataset):
    def __init__(self, manifest_path, prep: str, train: bool, jpeg_aligned=None, extra_transform=None):
        """jpeg_aligned overrides the prep token (eval.py uses False to test on untouched images).
        extra_transform is a PIL -> PIL perturbation applied before cropping (robustness tests)."""
        flags = prep_flags(prep)
        self.records = read_manifest(manifest_path)
        self.jpeg_aligned = flags["jpeg_aligned"] if jpeg_aligned is None else jpeg_aligned
        self.extra_transform = extra_transform
        self.transform = build_transform(flags["crop"], train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        with Image.open(rec["path"]) as im:
            is_jpeg = im.format == "JPEG"
            img = im.convert("RGB")
        label = float(rec["label"])
        if self.jpeg_aligned and label == 1.0 and not is_jpeg:
            img = jpeg_reencode(img)
        if self.extra_transform is not None:
            img = self.extra_transform(img)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def make_dataset(manifest_path: str, prep: str, train: bool) -> Dataset:
    return GenImageDataset(manifest_path, prep=prep, train=train)
