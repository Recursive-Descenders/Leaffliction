from pathlib import Path

import cv2  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np  # type: ignore[import-not-found]
from matplotlib.figure import Figure

from transformation.lesion_cache import LesionRecord, ensure_lesion_record
from transformation.leaf_cache import get_leaf_mask
from transformation.spot_mask import draw_lesion_contours
from transformation.util import (
    iter_image_paths,
    make_output_path,
    skip_image,
    validate_image_readable,
)

LESION_SUMMARY_ORDER = (
    "lesion_area_ratio",
    "lesion_count",
    "lesion_density",
    "avg_lesion_area",
    "lesion_area_std",
    "largest_lesion_area_ratio",
    "mean_lesion_solidity",
    "std_lesion_solidity",
    "mean_lesion_circularity",
    "std_lesion_circularity",
    "mean_border_complexity",
    "std_border_complexity",
)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _format_summary_text(summary: dict[str, float | int]) -> str:
    lines = ["Lesion features (from extraction)"]
    for key in LESION_SUMMARY_ORDER:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def build_lesion_analysis_figure(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    record: LesionRecord,
) -> Figure:
    contour_rgb = _bgr_to_rgb(
        draw_lesion_contours(image, record.spot_mask, leaf_mask)
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

    axes[0, 0].imshow(contour_rgb)
    axes[0, 0].set_title("Lesion contours")
    axes[0, 0].axis("off")

    table_ax = axes[0, 1]
    table_ax.axis("off")
    table_ax.set_title("Lesion summary")
    table_ax.text(
        0.02,
        0.98,
        _format_summary_text(record.lesion_summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=table_ax.transAxes,
    )

    figure.tight_layout()
    return figure


def lesion_analysis(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)
    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="lesion_analysis"):
        try:
            image = validate_image_readable(image_path)
        except ValueError as exc:
            skip_image(image_path, str(exc))
            continue

        record = get_leaf_mask(image_path)
        if record is None:
            skip_image(image_path, "no detected leaf")
            continue

        lesion_record = ensure_lesion_record(image_path)
        if lesion_record is None:
            skip_image(image_path, "no spot mask")
            continue

        figure = build_lesion_analysis_figure(
            image,
            record.mask,
            lesion_record,
        )
        output_file = make_output_path(
            src,
            dst,
            image_path,
            file,
            "lesion_analysis",
            extension=".png",
        )
        figure.savefig(output_file, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_file)

    return saved_paths
