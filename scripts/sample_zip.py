import argparse
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_subset import IMAGE_EXTS, PROTOCOLS, generator_of  # noqa: E402


def classify(name):
    """'X/imagenet_y/train/ai/a.png' -> (generator, 'train', 'ai'), or None if not a GenImage image."""
    p = PurePosixPath(name)
    if p.suffix.lower() not in IMAGE_EXTS or p.name.startswith(".") or len(p.parts) < 3:
        return None
    split, cls = p.parts[-3], p.parts[-2]
    if split not in ("train", "val") or cls not in ("ai", "nature"):
        return None
    gen = generator_of(PurePosixPath(*p.parts[:-3]))
    return (gen, split, cls) if gen else None


def safe_target(out: Path, name: str) -> Path:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"refusing unsafe path in zip: {name}")
    return out.joinpath(*p.parts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-only", action="store_true", help="print counts per folder, extract nothing")
    return p.parse_args()


def main():
    args = parse_args()
    train_gens = {g for proto in PROTOCOLS.values() for g in proto["train"]}
    for zpath in args.zip:
        with zipfile.ZipFile(zpath) as zf:
            groups = defaultdict(list)
            for info in zf.infolist():
                key = classify(info.filename)
                if key and key[1] in args.splits:
                    groups[key].append(info)
            print(f"[{zpath.name}]")
            for key, items in sorted(groups.items()):
                print(f"  {key}: {len(items)}")
            if args.list_only:
                continue

            todo = []
            for (gen, split, cls), items in sorted(groups.items()):
                n = args.n_test if split == "val" else (args.n_val + args.n_train if gen in train_gens else 0)
                if n == 0:
                    continue
                items = sorted(items, key=lambda i: i.filename)
                random.Random(f"{args.seed}-{gen}-{split}-{cls}").shuffle(items)
                if len(items) < n:
                    raise SystemExit(f"{(gen, split, cls)}: need {n}, zip has {len(items)}")
                todo += items[:n]

            written = 0
            for k, info in enumerate(todo, 1):
                target = safe_target(args.out, info.filename)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".part")
                    tmp.write_bytes(zf.read(info))  # zipfile verifies the CRC
                    tmp.rename(target)
                    written += 1
                if k % 2000 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} sampled ({written} new)", flush=True)


if __name__ == "__main__":
    main()
