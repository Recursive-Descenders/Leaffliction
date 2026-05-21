from __future__ import annotations

from pathlib import Path

import typer

from distribution.dataset import analyze_dataset
from distribution.visualization import create_visualizations, print_statistics


def run(
    directory_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing the dataset to analyze.",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        help="Save the generated visualization instead of showing it.",
    ),
    output_dir: Path | None = typer.Option(
        Path("outputs/distribution"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where the visualization should be saved "
        "(default: outputs/distribution).",
    ),
) -> None:
    """Analyze a dataset and visualize the class distribution."""
    plant_type = directory_path.name

    target_dir = output_dir if save else None
    if not save and output_dir != Path("outputs/distribution"):
        save = True

    disease_counts = analyze_dataset(str(directory_path))
    print_statistics(disease_counts)
    create_visualizations(disease_counts, plant_type,
                          target_dir if save else None)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
