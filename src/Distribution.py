import sys
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

def get_image_extensions() -> tuple:
    """Return supported image extensions."""
    return ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

def count_images_in_directory(directory: Path) -> int:
    """Count images in a directory."""
    count = 0
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in get_image_extensions():
            count += 1
    return count

def analyze_dataset(root_directory: str) -> Dict[str, int]:
    """
    Analyze the dataset structure and count images in each subdirectory.

    Args:
        root_directory: Path to the root directory containing disease subdirectories

    Returns:
        Dictionary with disease names as keys and image counts as values
    """
    root_path = Path(root_directory)

    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: Directory '{root_directory}' does not exist.")
        sys.exit(1)

    disease_counts: Dict[str, int] = {}

    # Check if this directory itself contains images (flat structure)
    direct_images = count_images_in_directory(root_path)
    if direct_images > 0:
        print(f"Found {direct_images} images directly in {root_path.name}/")
        disease_counts[root_path.name] = direct_images
        return disease_counts

    # Get all subdirectories
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"Error: No subdirectories found in '{root_directory}'.")
        sys.exit(1)

    for subdir in subdirs:
        disease_name = subdir.name
        image_count = count_images_in_directory(subdir)
        disease_counts[disease_name] = image_count

    return disease_counts

def create_visualizations(disease_counts: Dict[str, int], plant_type: str) -> None:
    """
    Create pie chart and bar chart for the disease distribution.

    Args:
        disease_counts: Dictionary with disease names and image counts
        plant_type: Name of the plant type (for chart titles)
    """
    if not disease_counts:
        print("No data to visualize.")
        return

    diseases = list(disease_counts.keys())
    counts = list(disease_counts.values())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{plant_type} class distribution', fontsize=14, fontweight='bold')

    # Pie chart
    colors = plt.cm.Set3(range(len(diseases)))
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=diseases,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('Distribution (%)', fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    # Bar chart
    bars = ax2.bar(diseases, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_title('Count', fontweight='bold')
    ax2.set_ylabel('Number of Images', fontweight='bold')
    ax2.set_xlabel('Disease Type', fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Rotate x labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def print_statistics(disease_counts: Dict[str, int]) -> None:
    """Print statistics about the dataset."""
    total_images = sum(disease_counts.values())
    print("\n" + "="*60)
    print(f"{'Disease Type':<30} {'Count':>10} {'Percentage':>10}")
    print("="*60)

    for disease, count in sorted(disease_counts.items()):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{disease:<30} {count:>10} {percentage:>9.1f}%")

    print("-"*60)
    print(f"{'Total':<30} {total_images:>10} {'100.0%':>10}")
    print("="*60 + "\n")

def main() -> None:
    """Main function to analyze dataset and create visualizations."""
    if len(sys.argv) != 2:
        print("Usage: uv run dist <directory_path>")
        print("Example: uv run dist images")
        sys.exit(1)

    directory_path = sys.argv[1]

    # Get plant type name from directory path
    plant_type = Path(directory_path).name

    # Analyze the dataset
    disease_counts = analyze_dataset(directory_path)

    # Print statistics
    print_statistics(disease_counts)

    # Create and display visualizations
    create_visualizations(disease_counts, plant_type)


if __name__ == "__main__":
    main()
