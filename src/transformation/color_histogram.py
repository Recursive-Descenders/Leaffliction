from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # type: ignore[import-not-found]

from transformation.color_features import (
    COLOR_CHART_COLORS,
    COLOR_NAMES,
    color_channel_histograms,
)
from transformation.leaf_cache import get_leaf_mask
from transformation.lesion_cache import ensure_lesion_record
from transformation.util import (
    iter_image_paths,
    make_output_path,
    skip_image,
    validate_image_readable,
)

CHANNEL_COLORS = dict(zip(COLOR_NAMES, COLOR_CHART_COLORS, strict=True))
SPOT_CHANNEL_COLORS = {
    f"spot_{name}": color
    for name, color in CHANNEL_COLORS.items()
}


def _plot_channel_histograms(
    axis: plt.Axes,
    histograms: dict[str, np.ndarray],
    *,
    title: str,
    colors: dict[str, str],
) -> None:
    x = np.arange(256)
    for label, proportions in histograms.items():
        axis.plot(
            x,
            proportions,
            label=label,
            color=colors.get(label, None),
            linewidth=1.5,
        )

    axis.set_title(title)
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Proportion of pixels")
    axis.set_xlim(0, 255)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Channel", fontsize=8)


def build_color_histogram_figure(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    spot_mask: np.ndarray,
) -> plt.Figure:
    leaf_histograms = color_channel_histograms(image, leaf_mask, spot=False)
    spot_histograms = color_channel_histograms(image, spot_mask, spot=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    _plot_channel_histograms(
        axes[0, 0],
        leaf_histograms,
        title="Leaf color histogram",
        colors=CHANNEL_COLORS,
    )
    _plot_channel_histograms(
        axes[0, 1],
        spot_histograms,
        title="Spot color histogram",
        colors=SPOT_CHANNEL_COLORS,
    )
    figure.tight_layout()
    return figure


def histogram(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)
    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="histogram"):
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

        figure = build_color_histogram_figure(
            image,
            record.mask,
            lesion_record.spot_mask,
        )
        output_file = make_output_path(
            src,
            dst,
            image_path,
            file,
            "histogram",
            extension=".png",
        )
        figure.savefig(output_file, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_file)

    return saved_paths
