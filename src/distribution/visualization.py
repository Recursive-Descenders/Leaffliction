from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def create_visualizations(
    disease_counts: Dict[str, int],
    plant_type: str,
    output_dir: Path | None = None,
) -> Path | None:
    """
    Create pie chart and bar chart for the disease distribution.

    Args:
        disease_counts: Dictionary with disease names and image counts
        plant_type: Name of the plant type (for chart titles)
    """
    if not disease_counts:
        print("No data to visualize.")
        return None

    diseases = list(disease_counts.keys())
    counts = list(disease_counts.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{plant_type} class distribution", fontsize=14, fontweight="bold")

    colors = plt.cm.Set3(range(len(diseases)))
    _, _, autotexts = ax1.pie(
        counts,
        labels=diseases,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax1.set_title("Distribution (%)", fontweight="bold")

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    bars = ax2.bar(diseases, counts, color=colors, edgecolor="black", linewidth=1.2)
    ax2.set_title("Count", fontweight="bold")
    ax2.set_ylabel("Number of Images", fontweight="bold")
    ax2.set_xlabel("Disease Type", fontweight="bold")

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.tick_params(axis="x", rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    output_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{plant_type}_class_distribution.png"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return output_path


def print_statistics(disease_counts: Dict[str, int]) -> None:
    """Print statistics about the dataset."""
    total_images = sum(disease_counts.values())
    print("\n" + "=" * 60)
    print(f"{'Disease Type':<30} {'Count':>10} {'Percentage':>10}")
    print("=" * 60)

    for disease, count in sorted(disease_counts.items()):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{disease:<30} {count:>10} {percentage:>9.1f}%")

    print("-" * 60)
    print(f"{'Total':<30} {total_images:>10} {'100.0%':>10}")
    print("=" * 60 + "\n")
