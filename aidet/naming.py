"""Run naming and preprocessing flags.

Team rule: one YAML config per run, named
    {protocol}_{model}_{init}_{prep}_lr{lr}_s{seed}
e.g. A_resnet50_in1k_raw-resize_lr1e-4_s0

Tokens may not contain "_" (it is the field separator); use "-" inside a token.
"""

# H4 2x2: JPEG-aligned vs raw fake training images, crop vs resize at input.
PREPS = ("raw-resize", "raw-crop", "jpeg-resize", "jpeg-crop")
INITS = ("in1k", "scratch")


def fmt_lr(lr: float) -> str:
    """0.0003 -> '3e-4', 1e-4 -> '1e-4', 1.5e-4 -> '1.5e-4'."""
    mantissa, exponent = f"{float(lr):e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def run_name(protocol, model, init, prep, lr, seed) -> str:
    for label, token in (("protocol", protocol), ("model", model), ("init", init), ("prep", prep)):
        if "_" in str(token):
            raise ValueError(f"{label} '{token}' may not contain '_' (use '-')")
    if prep not in PREPS:
        raise ValueError(f"prep '{prep}' not in {PREPS}")
    if init not in INITS:
        raise ValueError(f"init '{init}' not in {INITS}")
    return f"{protocol}_{model}_{init}_{prep}_lr{fmt_lr(lr)}_s{int(seed)}"


def run_name_from_cfg(cfg: dict) -> str:
    return run_name(cfg["protocol"], cfg["model"], cfg["init"], cfg["prep"], cfg["lr"], cfg["seed"])


def prep_flags(prep: str) -> dict:
    """Decode a prep token for the dataset: which JPEG alignment and input handling to use."""
    if prep not in PREPS:
        raise ValueError(f"prep '{prep}' not in {PREPS}")
    alignment, handling = prep.split("-")
    return {"jpeg_aligned": alignment == "jpeg", "crop": handling == "crop"}
