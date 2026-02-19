"""Prepare fold-based UrbanSound8K binary splits for gunshot detection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SplitSpec:
    """Fold IDs assigned to one output split."""

    name: str
    folds: tuple[int, ...]


SPLITS = (
    SplitSpec("train", (1, 2, 3, 4, 5, 6, 7, 8)),
    SplitSpec("val", (9,)),
    SplitSpec("test", (10,)),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create UrbanSound8K binary dataset using fold-based split."
    )
    parser.add_argument(
        "--urbansound_root",
        type=str,
        default=str(Path.home() / "Downloads" / "UrbanSound8K"),
        help="UrbanSound8K root containing metadata/ and audio/ folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_urbansound8k_binary",
        help="Output directory for train/val/test binary dataset.",
    )
    parser.add_argument(
        "--neg_to_pos_ratio",
        type=float,
        default=2.0,
        help="Maximum not_gunshot:gunshot ratio inside each split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return parser.parse_args()


def fold_to_split(fold: int) -> str:
    """Map UrbanSound fold number to output split name."""
    for spec in SPLITS:
        if fold in spec.folds:
            return spec.name
    raise ValueError(f"Unexpected fold: {fold}")


def load_rows(metadata_csv: Path, audio_root: Path) -> List[dict]:
    """Load metadata rows that have existing audio files on disk."""
    rows = []
    missing = 0
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fold = int(row["fold"])
            file_name = row["slice_file_name"]
            audio_path = audio_root / f"fold{fold}" / file_name
            if not audio_path.exists():
                missing += 1
                continue

            class_id = int(row["classID"])
            label = "gunshot" if class_id == 6 else "not_gunshot"
            rows.append(
                {
                    "split": fold_to_split(fold),
                    "fold": fold,
                    "file_name": file_name,
                    "audio_path": audio_path,
                    "label": label,
                }
            )
    print(f"Loaded {len(rows)} rows from metadata. Missing files: {missing}")
    return rows


def limit_negatives(rows: List[dict], ratio: float, seed: int) -> List[dict]:
    """Limit negatives per split to the configured ratio."""
    rng = random.Random(seed)
    kept: List[dict] = []
    for spec in SPLITS:
        split_rows = [r for r in rows if r["split"] == spec.name]
        positives = [r for r in split_rows if r["label"] == "gunshot"]
        negatives = [r for r in split_rows if r["label"] == "not_gunshot"]
        rng.shuffle(negatives)
        max_neg = int(len(positives) * ratio)
        selected_neg = negatives[: min(len(negatives), max_neg)]
        kept.extend(positives)
        kept.extend(selected_neg)
    return kept


def materialize(rows: List[dict], out_dir: Path, use_copy: bool) -> None:
    """Write split directories with symlinks or copies."""
    for row in rows:
        dest_dir = out_dir / row["split"] / row["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / row["file_name"]
        if dest_file.exists() or dest_file.is_symlink():
            continue
        if use_copy:
            shutil.copy2(row["audio_path"], dest_file)
        else:
            dest_file.symlink_to(row["audio_path"].resolve())


def write_manifest(rows: List[dict], out_dir: Path) -> None:
    """Write CSV manifest and JSON split stats."""
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "fold", "label", "file_name", "audio_path"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "fold": row["fold"],
                    "label": row["label"],
                    "file_name": row["file_name"],
                    "audio_path": str(row["audio_path"]),
                }
            )

    stats: Dict[str, Dict[str, int]] = {}
    for spec in SPLITS:
        split_rows = [r for r in rows if r["split"] == spec.name]
        pos = sum(1 for r in split_rows if r["label"] == "gunshot")
        neg = sum(1 for r in split_rows if r["label"] == "not_gunshot")
        stats[spec.name] = {"gunshot": pos, "not_gunshot": neg, "total": pos + neg}

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main() -> None:
    """Build dataset from UrbanSound8K metadata."""
    args = parse_args()
    if args.neg_to_pos_ratio <= 0:
        raise ValueError("--neg_to_pos_ratio must be > 0.")

    urbansound_root = Path(args.urbansound_root)
    metadata_csv = urbansound_root / "metadata" / "UrbanSound8K.csv"
    audio_root = urbansound_root / "audio"
    out_dir = Path(args.out_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_csv}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_root}")
    if out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}. Use a new --out_dir."
        )

    rows = load_rows(metadata_csv, audio_root)
    rows = limit_negatives(rows, ratio=args.neg_to_pos_ratio, seed=args.seed)
    materialize(rows, out_dir=out_dir, use_copy=args.copy)
    write_manifest(rows, out_dir=out_dir)
    mode = "copy" if args.copy else "symlink"
    print(f"Wrote dataset to {out_dir} ({mode} mode).")


if __name__ == "__main__":
    main()
