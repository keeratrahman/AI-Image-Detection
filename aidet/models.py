"""Model factory.

Contract for every model: forward(x: (B,3,224,224)) -> logits of shape (B,) or (B,1).
Label 1 = fake; train.py applies BCEWithLogitsLoss to the raw logit.

Teammates add their models with the registry, e.g. in this file:

    @register_model("deit-s")
    def _deit_s(init):
        ...  # return nn.Module, honoring init in {"in1k", "scratch"}

Model names may not contain "_" (see naming.py).
"""
import torch.nn as nn
import torchvision.models as tv

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def _resnet(ctor, weights_enum, init):
    if init not in ("in1k", "scratch"):
        raise ValueError(f"init must be 'in1k' or 'scratch', got '{init}'")
    model = ctor(weights=weights_enum if init == "in1k" else None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # single fake-vs-real logit
    return model


@register_model("resnet18")
def _resnet18(init):
    return _resnet(tv.resnet18, tv.ResNet18_Weights.IMAGENET1K_V1, init)


@register_model("resnet50")
def _resnet50(init):
    return _resnet(tv.resnet50, tv.ResNet50_Weights.IMAGENET1K_V1, init)


@register_model("deit-s")
def _deit_s(init):
    if init not in ("in1k", "scratch"):
        raise ValueError(f"init must be 'in1k' or 'scratch', got '{init}'")
    import timm

    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=init == "in1k",
        num_classes=1,
    )


class SmallCNN(nn.Module):
    """Four convolutional blocks and a pooled binary classification head."""

    def __init__(self):
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@register_model("cnn-small")
def _cnn_small(init):
    if init != "scratch":
        raise ValueError(f"cnn-small only supports init='scratch', got '{init}'")
    return SmallCNN()


def build_model(name: str, init: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](init)
