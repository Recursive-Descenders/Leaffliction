#!/usr/bin/env python3
"""
Augmentation CLI — unified entry, dispatches on input path type.

    # Single image → mode A: produce N K=2 augmented variants
    uv run aug "data/raw/Apple/apple_healthy/image (1).JPG" --n 6

    # Single class folder → mode B: fill to --target
    uv run aug data/train/Apple/apple_rust --target 1640

    # Dataset root (class sub-folders) → mode C: balance to max class
    uv run aug data/train/Apple --balance

Pipeline contract (decision graph):
- input is assumed already train/val-split — augmentation never touches val.
- mode B/C dedup byte-identical images before augmenting (decision #6).
- each output image is the composition of K=2 random methods from the pool
  (decision #1), applied at calibrated safe magnitudes (decision #2).
- output filenames encode the methods used (decision #9).
"""
from pathlib import Path

import typer

from augmentation.core import default_augmentor
from augmentation.preprocess import dedup_images, list_images
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
) -> None:
    """Mode A: produce ``n`` K=2 outputs from one image."""
    image = load_image(image_path)
    augmentor = default_augmentor(seed=seed)
    for i in range(n):
        result = augmentor.apply(image)
        out_path = build_aug_output_path(
            image_path, augmentor.last_methods, i, dst=dst
        )
        save_image(result, out_path)
        typer.echo(f"saved {out_path}")
    typer.echo(f"\n{n} augmented image(s) written to {dst}")


def _augment_class_folder(
    class_dir: Path,
    target: int,
    seed: int | None,
    dst: Path,
) -> tuple[int, int]:
    """Mode B core: fill one class folder to ``target`` via round-robin.

    Outputs land in ``dst / class_dir.name`` so the source tree is never
    mutated. Returns ``(produced, dropped_dups)``.
    """
    paths = list_images(class_dir)
    kept, dropped = dedup_images(paths)
    need = target - len(kept)
    if need <= 0:
        typer.echo(
            f"  {class_dir.name}: already at {len(kept)}/{target} — skip"
        )
        return 0, len(dropped)

    out_dir = dst / class_dir.name
    per_source = need // len(kept)
    remainder = need % len(kept)
    augmentor = default_augmentor(seed=seed)
    produced = 0
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
    return produced, len(dropped)


def _augment_dataset_root(
    data_dir: Path,
    target: int,
    balance: bool,
    seed: int | None,
    dst: Path,
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
            class_dir, target, seed, dst
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
        help="Single-image mode: number of K=2 outputs to produce.",
    ),
    target: int = typer.Option(
        0,
        "--target",
        "-t",
        help="Folder mode: target image count per class (0 = auto-detect).",
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
) -> None:
    """Augment input. Mode is decided by the input path type."""
    seed_value = seed if seed >= 0 else None

    if path.is_file():
        _augment_single_image(path, dst, n, seed_value)
        return

    # Directory: mode B if it directly contains images, else mode C.
    if _is_class_folder(path):
        if target <= 0:
            raise typer.BadParameter(
                "Class-folder mode requires --target N."
            )
        produced, dropped = _augment_class_folder(
            path, target, seed_value, dst
        )
        typer.echo(
            f"\ndone: +{produced} produced, -{dropped} duplicates dropped"
        )
        return

    _augment_dataset_root(path, target, balance, seed_value, dst)


def main() -> None:
    typer.run(augment)


if __name__ == "__main__":
    main()
