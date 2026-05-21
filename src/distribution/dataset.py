from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")


@dataclass(frozen=True)
class DatasetSummary:
    root_directory: Path
    disease_counts: dict[str, int]


def get_image_extensions() -> tuple[str, ...]:
    """Return supported image extensions."""
    return IMAGE_EXTENSIONS


def count_images_in_directory(directory: Path) -> int:
    """Count images in a directory."""
    count = 0
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in get_image_extensions():
            count += 1
    return count


def analyze_dataset(root_directory: str) -> dict[str, int]:
    """
    Analyze the dataset structure and count images in each subdirectory.

    Args:
        root_directory: Path to the root directory containing disease
                        subdirectories

    Returns:
        Dictionary with disease names as keys and image counts as values
    """
    root_path = Path(root_directory)

    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: Directory '{root_directory}' does not exist.")
        sys.exit(1)

    disease_counts: dict[str, int] = {}

    direct_images = count_images_in_directory(root_path)
    if direct_images > 0:
        print(f"Found {direct_images} images directly in {root_path.name}/")
        disease_counts[root_path.name] = direct_images
        return disease_counts

    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"Error: No subdirectories found in '{root_directory}'.")
        sys.exit(1)

    for subdir in subdirs:
        disease_counts[subdir.name] = count_images_in_directory(subdir)

    return disease_counts
