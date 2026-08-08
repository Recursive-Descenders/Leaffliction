#!/usr/bin/env python3
"""
Augmentation CLI — unified entry, dispatches on input path type.

    # Single image → mode A: produce N augmented variants
    uv run aug "data/raw/Apple/apple_healthy/image (1).JPG" --n 6

    # Single image with fixed methods
    uv run aug image.jpg --methods Flip,Rotate --n 4

    # Single class folder → mode B: fill to --target
    uv run aug data/train/Apple/apple_rust --target 1640

    # Dataset root (class sub-folders) → mode C: balance to max class
    uv run aug data/train/Apple --balance

    # Copy raw images alongside augmented output
    uv run aug data/train/Apple --balance --copy-raw --dst out/

Pipeline contract (decision graph):
- input is assumed already train/val-split — augmentation never touches val.
- mode B/C dedup byte-identical images before augmenting (decision #6).
- by default each output image is K=2 random methods from the pool (decision
  #1); use --methods to pin the sequence instead.
- output filenames encode the methods used (decision #9).
"""
import shutil
from pathlib import Path
from typing import Optional

import typer

from augmentation.core import default_augmentor
from augmentation.preprocess import dedup_images, list_images
from augmentation.registry import Method, lookup_methods
from augmentation.util import build_aug_output_path, load_image, save_image

DEFAULT_DST = Path("data/augmented_directory")
DEFAULT_N = 6
DEFAULT_SEED = 42


def _is_class_folder(path: Path) -> bool:
    """``path`` directly contains image files (no class sub-folders)."""
    return any(list_images(path))


def _augment_single_image(
    image_path: Path,
    dst: Path,
    n: int,
    seed: int | None,
    fixed_methods: list[Method] | None = None,
    copy_raw: bool = False,
) -> None:
    """Mode A: produce ``n`` augmented outputs from one image."""
    image = load_image(image_path)
    augmentor = default_augmentor(seed=seed, fixed_methods=fixed_methods)
    for i in range(n):
        result = augmentor.apply(image)
        out_path = build_aug_output_path(
            image_path, augmentor.last_methods, i, dst=dst
        )
        save_image(result, out_path)
        typer.echo(f"saved {out_path}")
    if copy_raw:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst / image_path.name)
        typer.echo(f"copied raw {image_path.name}")
    typer.echo(f"\n{n} augmented image(s) written to {dst}")


def _augment_class_folder(
    class_dir: Path,
    target: int,
    seed: int | None,
    dst: Path,
    fixed_methods: list[Method] | None = None,
    copy_raw: bool = False,
) -> tuple[int, int]:
    """Mode B core: fill one class folder to ``target`` via round-robin.

    Outputs land in ``dst / class_dir.name`` so the source tree is never
    mutated. Returns ``(produced, dropped_dups)``.
    """
    paths = list_images(class_dir)
    kept, dropped = dedup_images(paths)
    out_dir = dst / class_dir.name
    need = target - len(kept)

    produced = 0
    if need <= 0:
        typer.echo(
            f"  {class_dir.name}: already at {len(kept)}/{target} — skip aug"
        )
    else:
        per_source = need // len(kept)
        remainder = need % len(kept)
        augmentor = default_augmentor(seed=seed, fixed_methods=fixed_methods)
        for i, src in enumerate(kept):
            copies = per_source + (1 if i < remainder else 0)
            if copies == 0:
                continue
            image = load_image(src)
            for k in range(copies):
                result = augmentor.apply(image)
                out_path = build_aug_output_path(
                    src, augmentor.last_methods, k, dst=out_dir
                )
                save_image(result, out_path)
                produced += 1

        typer.echo(
            f"  {class_dir.name}: {len(kept)} src "
            f"(+{produced} aug, -{len(dropped)} dup) → {len(kept) + produced}"
        )

    if copy_raw:
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in kept:
            shutil.copy2(src, out_dir / src.name)

    return produced, len(dropped)


def _augment_dataset_root(
    data_dir: Path,
    target: int,
    balance: bool,
    seed: int | None,
    dst: Path,
    fixed_methods: list[Method] | None = None,
    copy_raw: bool = False,
) -> None:
    """Mode C: balance every class folder under ``data_dir``."""
    class_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and _is_class_folder(p)
    )
    if not class_dirs:
        raise typer.BadParameter(
            f"No images and no class sub-folders found under {data_dir}"
        )

    counts = {p.name: len(list_images(p)) for p in class_dirs}
    if balance or target <= 0:
        target = max(counts.values())
        typer.echo(f"auto-target = max class count = {target}")
    typer.echo(f"target per class = {target}\n")

    total_produced = 0
    total_dropped = 0
    for class_dir in class_dirs:
        produced, dropped = _augment_class_folder(
            class_dir, target, seed, dst,
            fixed_methods=fixed_methods,
            copy_raw=copy_raw,
        )
        total_produced += produced
        total_dropped += dropped

    typer.echo(
        f"\ndone: +{total_produced} produced, "
        f"-{total_dropped} duplicates dropped "
        f"across {len(class_dirs)} classes"
    )


def augment(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        resolve_path=True,
        help="Image file, class folder, or dataset root.",
    ),
    dst: Path = typer.Option(
        DEFAULT_DST,
        "--dst",
        "-d",
        help=(
            "Output directory. Folder modes write under "
            "<dst>/<class_name>/."
        ),
    ),
    n: int = typer.Option(
        DEFAULT_N,
        "--n",
        "-n",
        min=1,
        help="Single-image mode: number of augmented outputs to produce.",
    ),
    target: int = typer.Option(
        0,
        "--target",
        "-t",
        help=(
            "Folder mode: target augmented image count per class "
            "(0 = auto-detect)."
        ),
    ),
    balance: bool = typer.Option(
        False,
        "--balance",
        "-b",
        help="Dataset mode: auto-target the largest class count.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="RNG seed for reproducibility. Negative = stochastic.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        "-m",
        help=(
            "Comma-separated methods applied in order "
            "(e.g. --methods Flip,Rotate). "
            "Replaces random K=2 pool selection. "
            "Valid: Flip, Rotate, Shear, Skew, Crop, Distortion."
        ),
    ),
    copy_raw: bool = typer.Option(
        False,
        "--copy-raw",
        help=(
            "Copy source images into <dst>/<class>/ alongside augmented "
            "outputs. Raw copies are not counted toward --target."
        ),
    ),
) -> None:
    """Augment input. Mode is decided by the input path type."""
    seed_value = seed if seed >= 0 else None

    fixed = None
    if methods:
        names = [m.strip() for m in methods.split(",") if m.strip()]
        try:
            fixed = lookup_methods(names)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if path.is_file():
        _augment_single_image(
            path, dst, n, seed_value,
            fixed_methods=fixed,
            copy_raw=copy_raw,
        )
        return

    # Directory: mode B if it directly contains images, else mode C.
    if _is_class_folder(path):
        if target <= 0:
            raise typer.BadParameter(
                "Class-folder mode requires --target N."
            )
        produced, dropped = _augment_class_folder(
            path, target, seed_value, dst,
            fixed_methods=fixed,
            copy_raw=copy_raw,
        )
        typer.echo(
            f"\ndone: +{produced} produced, -{dropped} duplicates dropped"
        )
        return

    _augment_dataset_root(
        path, target, balance, seed_value, dst,
        fixed_methods=fixed,
        copy_raw=copy_raw,
    )


def main() -> None:
    typer.run(augment)


if __name__ == "__main__":
    main()
