from __future__ import annotations

import sys
from pathlib import Path

from distribution.dataset import analyze_dataset
from distribution.visualization import create_visualizations, print_statistics


def main() -> None:
    """Main function to analyze dataset and create visualizations."""
    if len(sys.argv) != 2:
        print("Usage: uv run dist <directory_path>")
        print("Example: uv run dist images")
        sys.exit(1)

    directory_path = sys.argv[1]
    plant_type = Path(directory_path).name

    disease_counts = analyze_dataset(directory_path)
    print_statistics(disease_counts)
    create_visualizations(disease_counts, plant_type)
