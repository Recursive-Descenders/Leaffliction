#!/usr/bin/env python3
"""
Augmentation CLI.

Takes a single image, applies every *implemented* geometric augmentation,
and writes each result next to the others in an output directory using the
subject naming rule ``<stem>_<Method><suffix>``.

Methods that are not implemented yet (they raise ``NotImplementedError``)
are skipped, so the set of outputs grows automatically as more transforms
land — no change needed here.

    uv run aug "data/raw/Apple/apple_healthy/image (1).JPG"
    ./src/Augmentation.py "data/raw/Apple/apple_healthy/image (1).JPG"
"""
from pathlib import Path

import typer

from augmentation.util import build_output_path, load_image, save_image
from augmentation.visualization import METHODS

DEFAULT_DST = Path("data/augmented_directory")


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
) -> None:
    """Apply every implemented augmentation and save the results."""
    image = load_image(image_path)

    saved: list[Path] = []
    for method in METHODS:
        defaults = {spec.name: spec.default for spec in method.params}
        try:
            result = method.func(image, **defaults)
        except NotImplementedError:
            typer.echo(f"skip  {method.label} (not implemented yet)")
            continue
        out_path = build_output_path(image_path, method.label, dst=dst)
        save_image(result, out_path)
        saved.append(out_path)
        typer.echo(f"saved {out_path}")

    typer.echo(f"\n{len(saved)} augmented image(s) written to {dst}")


def main() -> None:
    typer.run(augment)


if __name__ == "__main__":
    main()
