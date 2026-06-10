#!/usr/bin/env python3
"""
Augmentation CLI — mode A (single image).

Produces ``n`` augmented variants of a single source image by running it
through the K=2 Augmentor: each output is the composition of two random
methods drawn from the pool, applied at calibrated safe magnitudes.

    uv run aug "data/raw/Apple/apple_healthy/image (1).JPG"
    uv run aug "image.jpg" --n 5 --seed 42

The output filename encodes the two methods in call-order (decision #9):
``<stem>_<Method1>_<Method2>_<i><ext>``.
"""
from pathlib import Path

import typer

from augmentation.core import default_augmentor
from augmentation.util import build_aug_output_path, load_image, save_image

DEFAULT_DST = Path("data/augmented_directory")
DEFAULT_N = 6
DEFAULT_SEED = 42


def augment(
    image_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Source image to augment.",
    ),
    dst: Path = typer.Option(
        DEFAULT_DST,
        "--dst",
        "-d",
        help="Directory the augmented images are written to.",
    ),
    n: int = typer.Option(
        DEFAULT_N,
        "--n",
        "-n",
        min=1,
        help="Number of K=2 augmented outputs to produce.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="RNG seed for reproducibility. Negative = stochastic run.",
    ),
) -> None:
    """Produce ``n`` K=2 augmented variants of the source image."""
    image = load_image(image_path)
    augmentor = default_augmentor(seed=seed if seed >= 0 else None)

    for i in range(n):
        result = augmentor.apply(image)
        out_path = build_aug_output_path(
            image_path, augmentor.last_methods, i, dst=dst
        )
        save_image(result, out_path)
        typer.echo(f"saved {out_path}")

    typer.echo(f"\n{n} augmented image(s) written to {dst}")


def main() -> None:
    typer.run(augment)


if __name__ == "__main__":
    main()
