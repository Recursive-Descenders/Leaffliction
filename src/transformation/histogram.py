from pathlib import Path
import altair as alt
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd
from plantcv.plantcv import print_image  # type: ignore[import-not-found]
from plantcv.plantcv.visualize.histogram import (  # type: ignore
    histogram as compute_channel_histogram,
)
from transformation.mask import build_mask

from transformation.util import (
    iter_image_paths,
    largest_leaf_mask,
    make_output_path,
)

CHANNEL_ORDER = [
    "blue",
    "blue-yellow",
    "green",
    "green-magenta",
    "hue",
    "lightness",
    "red",
    "saturation",
    "value",
]

CHANNEL_COLORS = [
    "blue",
    "yellow",
    "forestgreen",
    "magenta",
    "blueviolet",
    "dimgray",
    "red",
    "cyan",
    "orange",
]


def _color_channels(
    image: np.ndarray,
    leaf_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    masked = cv2.bitwise_and(image, image, mask=leaf_mask)
    blue, green, red = cv2.split(masked)
    lightness, green_magenta, blue_yellow = cv2.split(
        cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    )
    hue, saturation, value = cv2.split(
        cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    )
    return {
        "blue": blue,
        "blue-yellow": blue_yellow,
        "green": green,
        "green-magenta": green_magenta,
        "hue": hue,
        "lightness": lightness,
        "red": red,
        "saturation": saturation,
        "value": value,
    }


def _build_histogram_chart(
    leaf_mask: np.ndarray,
    channels: dict[str, np.ndarray],
) -> alt.Chart:
    parts: list[pd.DataFrame] = []

    for name in CHANNEL_ORDER:
        _, hist_df = compute_channel_histogram(
            channels[name],
            mask=leaf_mask,
            bins=256,
            lower_bound=0,
            upper_bound=255,
            hist_data=True,
        )
        part = pd.DataFrame(
            {
                "pixel intensity": hist_df["pixel intensity"],
                "proportion of pixels (%)": hist_df[
                    "proportion of pixels (%)"
                ],
                "channel": name,
            }
        )
        parts.append(part)

    data = pd.concat(parts, ignore_index=True)

    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            alt.X("pixel intensity:Q", title="Pixel intensity"),
            alt.Y(
                "proportion of pixels (%):Q",
                title="Proportion of pixels (%)",
            ),
            color=alt.Color(
                "channel:N",
                title="Channel",
                scale=alt.Scale(
                    domain=CHANNEL_ORDER,
                    range=CHANNEL_COLORS,
                ),
            ),
        )
        .properties(width=700, height=400)
    )


def build_histogram_chart(image: np.ndarray) -> alt.Chart | None:
    leaf_mask = largest_leaf_mask(build_mask(image))
    if leaf_mask is None:
        return None

    channels = _color_channels(image, leaf_mask)
    return _build_histogram_chart(leaf_mask, channels)


def histogram(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="histogram"):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        chart = build_histogram_chart(image)
        if chart is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        output_file = make_output_path(
            src, dst, image_path, file, "histogram"
        ).with_suffix(".png")
        print_image(chart, str(output_file))
        saved_paths.append(output_file)

    return saved_paths
