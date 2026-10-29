import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

# First matching pattern wins; matched against the lowercased folder path.
GENERATOR_PATTERNS = [
    ("sdv14", r"sdv4|v_?1[._]?4|sd_?1[._]?4"),
    ("sdv15", r"sdv5|v_?1[._]?5|sd_?1[._]?5"),
    ("midjourney", r"midjourney"),
    ("wukong", r"wukong"),
    ("biggan", r"biggan"),
    ("glide", r"glide"),
    ("vqdm", r"vqdm"),
    ("adm", r"adm"),
]
GENERATORS = [g for g, _ in GENERATOR_PATTERNS]

# Fixed protocols (see docs/PLAN.md). Everything not in "train" is held out.
PROTOCOLS = {
    "A": {"train": ["sdv14"]},                             # GenImage standard: train SD v1.4
    "B": {"train": ["sdv14", "wukong", "glide", "vqdm"]},  # multi-generator training
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def generator_of(path: Path):
    s = str(path).lower()
    for name, pattern in GENERATOR_PATTERNS:
        if re.search(pattern, s):
            return name
    return None


def list_images(folder: Path):
    # Skip macOS "._" AppleDouble files, which exFAT/FAT drives create next to every file.
    return sorted(folder / name for name in os.listdir(folder)
                  if Path(name).suffix.lower() in IMAGE_EXTS and not name.startswith("."))


def discover(src: Path, splits=("train", "val")):
    """-> {(generator, split, cls): [paths]} for split in `splits`, cls in {ai, nature}.

    Walks directories only and never descends into image folders, so it stays fast on
    network filesystems (e.g. Kaggle's /kaggle/input) with hundreds of thousands of files.
    The generator is read from the path including src's own name, so --src may be the
    generator folder itself (e.g. .../stable_diffusion_v_1_4)."""
    src = src.resolve()
    found = defaultdict(list)
    for dirpath, dirnames, _ in os.walk(src):
        here = Path(dirpath)
        if here.name in ("train", "val"):
            for cls in ("ai", "nature"):
                if cls in dirnames and here.name in splits:
                    gen = generator_of((here / cls).relative_to(src.parent))
                    if gen is None:
                        print(f"  ! skipping {here / cls}: cannot tell which generator it is")
                    else:
                        found[(gen, here.name, cls)] += list_images(here / cls)
            dirnames[:] = [d for d in dirnames if d not in ("ai", "nature")]
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return found


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--n-train", type=int, default=6000,
                   help="total fake (and total real) training images per protocol")
    p.add_argument("--n-val", type=int, default=1000, help="fake (and real) val images per training generator")
    p.add_argument("--n-test", type=int, default=1000, help="fake (and real) test images per generator")
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"],
                   help="only use these GenImage splits from --src (e.g. train when test images come from elsewhere)")
    p.add_argument("--dry-run", action="store_true", help="print what was found, copy nothing")
    p.add_argument("--pool-only", action="store_true",
                   help="only copy a random sample of whatever generators are in --src into <dest>/images "
                        "(no manifests); for downloading one generator at a time")
    return p.parse_args()


def main():
    args = parse_args()
    found = discover(args.src, args.splits)
    print("Found (generator, split, class): count")
    for key in sorted(found):
        print(f"  {key}: {len(found[key])}")
    if args.dry_run:
        return

    # Shuffle each folder once with a fixed seed; every manifest takes a prefix of that order,
    # so val and train never overlap and protocols share images where they share generators.
    order, need = {}, defaultdict(int)

    def take(gen, split, cls, start, n):
        key = (gen, split, cls)
        if key not in order:
            order[key] = list(found.get(key, []))
            random.Random(f"{args.seed}-{gen}-{split}-{cls}").shuffle(order[key])
        if len(order[key]) < start + n:
            raise SystemExit(f"{key}: need {start + n} images, only {len(order[key])} available")
        need[key] = max(need[key], start + n)
        return [(p, gen, split, cls) for p in order[key][start:start + n]]

    if args.pool_only:
        train_gens = {g for proto in PROTOCOLS.values() for g in proto["train"]}
        for gen, split, cls in list(found):
            if split == "val":
                take(gen, split, cls, 0, args.n_test)
            elif gen in train_gens:
                take(gen, split, cls, 0, args.n_val + args.n_train)

    manifests = {}
    for gen in GENERATORS if not args.pool_only else []:
        if (gen, "val", "ai") in found:
            manifests[f"test_{gen}"] = take(gen, "val", "ai", 0, args.n_test) + take(gen, "val", "nature", 0, args.n_test)

    for proto in args.protocols if not args.pool_only else []:
        train_gens = PROTOCOLS[proto]["train"]
        missing = [g for g in train_gens if (g, "train", "ai") not in found]
        if missing:
            print(f"  ! protocol {proto}: no train data for {missing}, skipping")
            continue
        per_gen = args.n_train // len(train_gens)
        manifests[f"{proto}_val"] = [x for g in train_gens for c in ("ai", "nature")
                                     for x in take(g, "train", c, 0, args.n_val)]
        manifests[f"{proto}_train"] = [x for g in train_gens for c in ("ai", "nature")
                                       for x in take(g, "train", c, args.n_val, per_gen)]

    # Copy files (copyfile, not copy2: no metadata, so no "._" sidecar files on exFAT).
    rel = {}
    for key, n in need.items():
        gen, split, cls = key
        d = args.dest / "images" / gen / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for p in order[key][:n]:
            target = d / p.name
            if not target.exists():
                shutil.copyfile(p, target)
            rel[p] = target.relative_to(args.dest).as_posix()
    print(f"{len(rel)} sampled images in {args.dest / 'images'}")

    if args.pool_only:
        print(f"Pool updated. When every generator is pooled, build manifests with\n"
              f"  --src {args.dest / 'images'} --dest {args.dest}")
        return
    rng = random.Random(args.seed)
    mdir = args.dest / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, items in manifests.items():
        rng.shuffle(items)
        with open(mdir / f"{name}.jsonl", "w") as f:
            for p, gen, split, c in items:
                f.write(json.dumps({"path": rel[p], "label": int(c == "ai"),
                                    "generator": gen, "split": split}) + "\n")
        n_fake = sum(c == "ai" for *_, c in items)
        print(f"  {name}.jsonl: {len(items)} images ({n_fake} fake / {len(items) - n_fake} real)")
    with open(mdir / "subset_info.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "protocols": PROTOCOLS}, f, indent=2)


if __name__ == "__main__":
    main()
