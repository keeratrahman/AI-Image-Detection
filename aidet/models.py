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


def build_model(name: str, init: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](init)
