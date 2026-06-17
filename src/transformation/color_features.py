import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

COLOR_NAMES = ("green", "yellow", "orange", "brown", "dark")

COLOR_CHART_COLORS = (
    "forestgreen",
    "gold",
    "darkorange",
    "saddlebrown",
    "dimgray",
)


def color_category_masks(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "green": (
            (hue >= 35)
            & (hue <= 95)
            & (saturation >= 35)
            & (value >= 35)
        ),
        "yellow": (
            (hue >= 20)
            & (hue < 35)
            & (saturation >= 40)
            & (value >= 40)
        ),
        "orange": (
            (hue >= 10)
            & (hue < 20)
            & (saturation >= 45)
            & (value >= 35)
        ),
        "brown": (
            (hue <= 25)
            & (saturation >= 40)
            & (value >= 15)
            & (value < 110)
        ),
        "dark": value < 45,
    }


def _masked_hsv_pixels(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    masked_pixels = mask > 0
    if not np.any(masked_pixels):
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    return (
        hue[masked_pixels],
        saturation[masked_pixels],
        value[masked_pixels],
    )


def color_ratios(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    pixels = _masked_hsv_pixels(image, mask)
    if pixels is None:
        return {name: 0.0 for name in COLOR_NAMES}

    hue, saturation, value = pixels
    pixel_count = len(hue)
    categories = color_category_masks(hue, saturation, value)

    return {
        name: float(np.count_nonzero(categories[name]) / pixel_count)
        for name in COLOR_NAMES
    }


def hsv_stats(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    pixels = _masked_hsv_pixels(image, mask)
    if pixels is None:
        return {
            "saturation_mean": 0.0,
            "saturation_std": 0.0,
            "value_mean": 0.0,
            "value_std": 0.0,
            "color_variance": 0.0,
        }

    hue, saturation, value = pixels
    return {
        "saturation_mean": float(np.mean(saturation)),
        "saturation_std": float(np.std(saturation)),
        "value_mean": float(np.mean(value)),
        "value_std": float(np.std(value)),
        "color_variance": float(np.var(hue.astype(np.float64))),
    }


def _channel_label(color: str, *, spot: bool) -> str:
    return f"spot_{color}" if spot else color


def color_channel_histograms(
    image: np.ndarray,
    region_mask: np.ndarray,
    *,
    spot: bool = False,
) -> dict[str, np.ndarray]:
    if not np.any(region_mask > 0):
        return {
            _channel_label(name, spot=spot): np.zeros(256, dtype=np.float64)
            for name in COLOR_NAMES
        }

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    in_region = region_mask > 0
    categories = color_category_masks(hue, saturation, value)

    histograms: dict[str, np.ndarray] = {}
    for name in COLOR_NAMES:
        channel_pixels = hue[in_region & categories[name]]
        if channel_pixels.size == 0:
            histograms[_channel_label(name, spot=spot)] = np.zeros(
                256,
                dtype=np.float64,
            )
            continue

        counts, _ = np.histogram(
            channel_pixels,
            bins=256,
            range=(0, 256),
        )
        proportions = counts.astype(np.float64)
        total = proportions.sum()
        if total > 0:
            proportions /= total
        histograms[_channel_label(name, spot=spot)] = proportions

    return histograms
