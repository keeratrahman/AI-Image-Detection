"""Datasets and loaders.

Contract with the real data pipeline (proposed to Nhi; adjust together):

    aidet/genimage.py must expose
        make_dataset(manifest_path: str, prep: str, train: bool) -> torch.utils.data.Dataset
    where each item is (image, label):
        image: float32 tensor (3, 224, 224), ImageNet-normalized
        label: float32 scalar tensor, 1.0 = fake, 0.0 = real
    `prep` is one of naming.PREPS; use naming.prep_flags(prep) to decode JPEG alignment and
    crop-vs-resize. Training crops randomly at native resolution; never resize when crop=True.

Until that exists, data.name: stub in the config uses StubDataset (random tensors).
"""
import torch
from torch.utils.data import DataLoader, Dataset

from aidet.naming import prep_flags

IMAGE_SIZE = 224


class StubDataset(Dataset):
    """Random 3x224x224 tensors. Fakes (label 1) carry a fixed checkerboard so a working
    training loop can actually learn something; this is only for pipeline smoke tests."""

    def __init__(self, size: int, seed: int = 0, image_size: int = IMAGE_SIZE):
        self.size = size
        self.seed = seed
        self.image_size = image_size
        idx = torch.arange(image_size)
        checker = ((idx[:, None] + idx[None, :]) % 2).float() * 2 - 1
        self.pattern = 0.5 * checker.unsqueeze(0)  # (1, H, W)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        label = float(i % 2)
        img = 0.5 * torch.randn(3, self.image_size, self.image_size, generator=g)
        if label == 1.0:
            img = img + self.pattern
        return img, torch.tensor(label, dtype=torch.float32)


def build_datasets(cfg: dict):
    d = cfg["data"]
    name = d["name"]
    if name == "stub":
        train_ds = StubDataset(d.get("stub_train_size", 128), seed=cfg["seed"])
        val_ds = StubDataset(d.get("stub_val_size", 64), seed=10_000 + cfg["seed"])
        return train_ds, val_ds
    if name == "genimage":
        prep_flags(cfg["prep"])  # fail early on a bad prep token
        try:
            from aidet.genimage import make_dataset
        except ImportError as e:
            raise NotImplementedError(
                "aidet/genimage.py (make_dataset) is not merged yet; see the contract in aidet/data.py"
            ) from e
        train_ds = make_dataset(d["train_manifest"], prep=cfg["prep"], train=True)
        val_ds = make_dataset(d["val_manifest"], prep=cfg["prep"], train=False)
        return train_ds, val_ds
    raise ValueError(f"Unknown data.name '{name}' (expected 'stub' or 'genimage')")


def build_dataloaders(cfg: dict, device: torch.device):
    train_ds, val_ds = build_datasets(cfg)
    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) >= bs,
        num_workers=nw, pin_memory=pin, generator=g, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader
