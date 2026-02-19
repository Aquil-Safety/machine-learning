"""Create balanced train/val splits for Aquili audio data."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List


AUDIO_EXTENSIONS = {".wav", ".wave", ".mp3", ".flac"}
CLASSES = ("gunshot", "not_gunshot")
SOURCE_SPLITS = ("train", "val", "test")


def list_audio_files(folder: Path) -> List[Path]:
    """Return all supported audio files under a folder."""
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )


def collect_pool(data_dir: Path) -> Dict[str, List[Path]]:
    """Collect class files from train/val/test into one source pool."""
    pool: Dict[str, List[Path]] = {name: [] for name in CLASSES}
    for split in SOURCE_SPLITS:
        for class_name in CLASSES:
            class_dir = data_dir / split / class_name
            pool[class_name].extend(list_audio_files(class_dir))

    for class_name in CLASSES:
        unique = sorted(set(pool[class_name]))
        pool[class_name] = unique
    return pool


def pick_balanced_subset(
    pool: Dict[str, List[Path]],
    max_neg_to_pos_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Path]]:
    """Downsample negatives to keep class imbalance within the configured ratio."""
    pos = list(pool["gunshot"])
    neg = list(pool["not_gunshot"])

    if not pos or not neg:
        raise RuntimeError(
            "Both classes need at least one file. "
            f"Found gunshot={len(pos)}, not_gunshot={len(neg)}."
        )

    rng.shuffle(pos)
    rng.shuffle(neg)

    max_neg = max(1, int(len(pos) * max_neg_to_pos_ratio))
    neg = neg[: min(len(neg), max_neg)]

    return {"gunshot": pos, "not_gunshot": neg}


def split_class_files(
    files: List[Path],
    val_ratio: float,
    min_val_count: int,
    rng: random.Random,
) -> Dict[str, List[Path]]:
    """Split one class into train/val with a minimum validation count."""
    files = list(files)
    rng.shuffle(files)

    val_count = max(min_val_count, int(round(len(files) * val_ratio)))
    val_count = min(max(1, val_count), len(files) - 1)

    return {"val": files[:val_count], "train": files[val_count:]}


def target_name(src: Path) -> str:
    """Create a collision-resistant output filename."""
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{src.stem}_{digest}{src.suffix.lower()}"


def copy_splits(
    split_map: Dict[str, Dict[str, List[Path]]],
    out_dir: Path,
    use_symlinks: bool,
) -> None:
    """Materialize split files in out_dir."""
    for split in ("train", "val"):
        for class_name in CLASSES:
            dest_dir = out_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_map[split][class_name]:
                dst = dest_dir / target_name(src)
                if use_symlinks:
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)


def print_summary(title: str, split_map: Dict[str, Dict[str, List[Path]]]) -> None:
    """Print split counts in a compact format."""
    print(title)
    for split in ("train", "val"):
        pos = len(split_map[split]["gunshot"])
        neg = len(split_map[split]["not_gunshot"])
        total = pos + neg
        ratio = (neg / pos) if pos else float("inf")
        print(
            f"  {split}: total={total}, gunshot={pos}, "
            f"not_gunshot={neg}, neg:pos={ratio:.2f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Create balanced train/val splits from local audio pool.")
    parser.add_argument("--data_dir", type=str, default="data", help="Input dataset root.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_rebalanced",
        help="Output dataset root for generated splits.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio per class.")
    parser.add_argument(
        "--max_neg_to_pos_ratio",
        type=float,
        default=2.0,
        help="Maximum not_gunshot:gunshot ratio after balancing.",
    )
    parser.add_argument(
        "--min_val_count",
        type=int,
        default=5,
        help="Minimum validation samples per class when possible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of file copies.")
    parser.add_argument("--dry_run", action="store_true", help="Preview split counts without writing files.")
    return parser.parse_args()


def main() -> None:
    """Build balanced splits and optionally materialize them to disk."""
    args = parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1.")
    if args.max_neg_to_pos_ratio <= 0:
        raise ValueError("--max_neg_to_pos_ratio must be > 0.")
    if args.min_val_count < 1:
        raise ValueError("--min_val_count must be >= 1.")

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    pool = collect_pool(data_dir)
    print(
        "Source pool: "
        f"gunshot={len(pool['gunshot'])}, not_gunshot={len(pool['not_gunshot'])}"
    )

    balanced = pick_balanced_subset(pool, args.max_neg_to_pos_ratio, rng)

    gun_split = split_class_files(
        balanced["gunshot"], args.val_ratio, args.min_val_count, rng
    )
    neg_split = split_class_files(
        balanced["not_gunshot"], args.val_ratio, args.min_val_count, rng
    )

    split_map = {
        "train": {"gunshot": gun_split["train"], "not_gunshot": neg_split["train"]},
        "val": {"gunshot": gun_split["val"], "not_gunshot": neg_split["val"]},
    }
    print_summary("Planned split:", split_map)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    if out_dir.exists():
        raise RuntimeError(
            f"Output directory already exists: {out_dir}. "
            "Remove it or pass a different --out_dir."
        )

    copy_splits(split_map, out_dir, use_symlinks=args.symlink)
    print(f"Wrote rebalanced dataset to: {out_dir}")
    mode = "symlinks" if args.symlink else "copied files"
    print(f"Materialization mode: {mode}")


if __name__ == "__main__":
    main()
