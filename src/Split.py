#!/usr/bin/env python3
"""
Split CLI (decision #7) — stratified per-class train/val split.

Splits each class folder under ``DATA_DIR`` into ``train/`` and ``val/``
output directories using stratified random sampling, so the per-class
ratio is preserved in both splits. Files are *copied*, not moved.

Run BEFORE ``aug-dataset``. The augmentor is supposed to only see
``train/``; ``val/`` must remain pure real images.

    uv run split data/raw/Apple --val 0.2 --seed 42
    # → data/split/train/Apple/apple_healthy/, data/split/val/...

Layout assumption (matches subject convention)::

    DATA_DIR/
      class_a/
        image (1).jpg
        ...
      class_b/
        ...
"""
import shutil
from pathlib import Path

import numpy as np
import typer

from augmentation.preprocess import list_images

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42


def _split_class(
    class_dir: Path,
    rng: np.random.Generator,
    val_fraction: float,
    train_root: Path,
    val_root: Path,
) -> tuple[int, int]:
    images = list_images(class_dir)
    n = len(images)
    if n == 0:
        return 0, 0

    indices = rng.permutation(n)
    val_count = max(1, int(round(n * val_fraction)))
    val_idx = set(indices[:val_count].tolist())

    train_dir = train_root / class_dir.name
    val_dir = val_root / class_dir.name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    for i, src in enumerate(images):
        dst_dir = val_dir if i in val_idx else train_dir
        shutil.copy2(src, dst_dir / src.name)
        if dst_dir is train_dir:
            train_count += 1

    return train_count, val_count


def split(
    data_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
        help="Dataset root containing class sub-folders.",
    ),
    out: Path = typer.Option(
        Path("data/split"),
        "--out",
        "-o",
        help="Root directory for train/ and val/ outputs.",
    ),
    val_fraction: float = typer.Option(
        DEFAULT_VAL_FRACTION,
        "--val",
        "-v",
        min=0.0,
        max=0.9,
        help="Fraction of each class to put in the val split.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="RNG seed for reproducibility. Negative = stochastic run.",
    ),
) -> None:
    """Stratified per-class train/val split (copies files)."""
    class_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and any(list_images(p))
    )
    if not class_dirs:
        raise typer.BadParameter(f"No class folders found under {data_dir}")

    rng = np.random.default_rng(seed if seed >= 0 else None)
    train_root = out / "train"
    val_root = out / "val"

    total_train = 0
    total_val = 0
    for class_dir in class_dirs:
        t, v = _split_class(class_dir, rng, val_fraction, train_root, val_root)
        typer.echo(f"  {class_dir.name}: train={t} val={v}")
        total_train += t
        total_val += v

    typer.echo(
        f"\ndone: train={total_train}  val={total_val}  → {out}"
    )


def main() -> None:
    typer.run(split)


if __name__ == "__main__":
    main()
