#!/usr/bin/env python3
"""
Augmentation CLI — mode B/C (class folder / dataset root).

Balances class folders by augmenting minority classes up to the largest
class count (decision #4 — match max), using round-robin source sampling
(decision #5) on top of byte-hash dedup (decision #6).

    # Mode C — full dataset
    uv run aug-dataset data/train/Apple --balance --seed 42

    # Mode B — single class folder (point at the folder directly)
    uv run aug-dataset data/train/Apple/apple_rust --target 1640

Layout assumption (matches subject convention)::

    DATA_DIR/
      class_a/
        image (1).jpg
        ...
      class_b/
        ...

If ``DATA_DIR`` *is* a class folder (contains images directly, not
sub-folders), mode B kicks in and you must pass ``--target``.
"""
from pathlib import Path

import typer

from augmentation.core import default_augmentor
from augmentation.preprocess import dedup_images, list_images
from augmentation.util import build_aug_output_path, load_image, save_image

DEFAULT_SEED = 42


def _is_class_folder(path: Path) -> bool:
    return any(list_images(path))


def _augment_class(
    class_dir: Path,
    target: int,
    seed: int,
) -> tuple[int, int]:
    """Augment one class folder up to ``target``.

    Returns ``(produced, dropped_dups)`` for logging.
    """
    paths = list_images(class_dir)
    kept, dropped = dedup_images(paths)

    need = target - len(kept)
    if need <= 0:
        typer.echo(
            f"  {class_dir.name}: already at {len(kept)}/{target} — skipping"
        )
        return 0, len(dropped)

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
                src, augmentor.last_methods, k, dst=class_dir
            )
            save_image(result, out_path)
            produced += 1

    typer.echo(
        f"  {class_dir.name}: {len(kept)} src "
        f"(+{produced} aug, -{len(dropped)} dup) → {len(kept) + produced}"
    )
    return produced, len(dropped)


def augment_dataset(
    data_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
        help="Dataset root (class sub-folders) or a single class folder.",
    ),
    target: int = typer.Option(
        0,
        "--target",
        "-t",
        help="Target image count per class. 0 = auto (max class count).",
    ),
    balance: bool = typer.Option(
        False,
        "--balance",
        "-b",
        help="Auto-target the largest class (overrides --target).",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="RNG seed for reproducibility. Negative = stochastic run.",
    ),
) -> None:
    """Augment classes up to a target count (round-robin)."""
    seed_value = seed if seed >= 0 else None

    # Mode B: pointed at a single class folder.
    if _is_class_folder(data_dir):
        if target <= 0:
            raise typer.BadParameter(
                "Mode B (single class folder) requires --target."
            )
        produced, dropped = _augment_class(data_dir, target, seed_value)
        typer.echo(
            f"\ndone: +{produced} produced, -{dropped} duplicates dropped"
        )
        return

    # Mode C: dataset root — discover class folders.
    class_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and _is_class_folder(p)
    )
    if not class_dirs:
        raise typer.BadParameter(
            f"No class folders found under {data_dir}"
        )

    # Compute target.
    counts = {p.name: len(list_images(p)) for p in class_dirs}
    if balance or target <= 0:
        target = max(counts.values())
        typer.echo(f"auto-target = max class count = {target}")
    typer.echo(f"target per class = {target}\n")

    total_produced = 0
    total_dropped = 0
    for class_dir in class_dirs:
        produced, dropped = _augment_class(class_dir, target, seed_value)
        total_produced += produced
        total_dropped += dropped

    typer.echo(
        f"\ndone: +{total_produced} produced, "
        f"-{total_dropped} duplicates dropped across {len(class_dirs)} classes"
    )


def main() -> None:
    typer.run(augment_dataset)


if __name__ == "__main__":
    main()
