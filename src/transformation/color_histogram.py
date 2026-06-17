from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # type: ignore[import-not-found]

from transformation.color_features import (
    COLOR_CHART_COLORS,
    COLOR_NAMES,
    color_channel_histograms,
)
from transformation.leaf_cache import get_leaf_mask
from transformation.lesion_cache import ensure_lesion_record
from transformation.parallel import parallel_map
from transformation.util import (
    get_image_paths,
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


@dataclass(frozen=True)
class HistogramWorkItem:
    src: str
    dst: str
    file: str | None
    image_path: str


def process_histogram_image(item: HistogramWorkItem) -> str | None:
    matplotlib.use("Agg")

    src = Path(item.src)
    dst = Path(item.dst)
    image_path = Path(item.image_path)
    file = item.file

    try:
        image = validate_image_readable(image_path)
    except ValueError as exc:
        skip_image(image_path, str(exc))
        return None

    record = get_leaf_mask(image_path, image=image)
    if record is None:
        skip_image(image_path, "no detected leaf")
        return None

    lesion_record = ensure_lesion_record(image_path, image=image)
    if lesion_record is None:
        skip_image(image_path, "no spot mask")
        return None

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
    return str(output_file)


def histogram(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
    *,
    jobs: int = 0,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)
    file_name = str(file) if file is not None else None

    items = [
        HistogramWorkItem(
            src=str(src),
            dst=str(dst),
            file=file_name,
            image_path=str(image_path),
        )
        for image_path in get_image_paths(src, file)
    ]
    results = parallel_map(
        process_histogram_image,
        items,
        jobs=jobs,
        desc="histogram",
    )
    return [Path(path) for path in results if path is not None]
