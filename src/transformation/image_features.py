from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from transformation.color_features import COLOR_NAMES, color_ratios, hsv_stats


@dataclass(frozen=True)
class LesionMeasurement:
    label_id: int
    area: float
    perimeter: float
    circularity: float
    solidity: float
    centroid: tuple[float, float]


@dataclass(frozen=True)
class ImageColorFeatures:
    leaf_ratios: dict[str, float]
    spot_ratios: dict[str, float]
    leaf_stats: dict[str, float]
    spot_stats: dict[str, float]


@dataclass(frozen=True)
class ImageFeatures:
    color: ImageColorFeatures
    lesions: tuple[LesionMeasurement, ...]
    lesion_summary: dict[str, float | int]


def _color_features_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    ratios = color_ratios(image, mask)
    stats = hsv_stats(image, mask)

    features: dict[str, float] = {
        f"{prefix}{name}_ratio": ratios[name]
        for name in COLOR_NAMES
    }
    features.update({
        f"{prefix}{key}": value
        for key, value in stats.items()
    })
    return features


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return 0.0, 0.0
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _measure_lesions(
    spot_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> tuple[tuple[LesionMeasurement, ...], dict[str, float | int]]:
    leaf_area = max(int(np.count_nonzero(leaf_mask)), 1)
    lesion_pixels = int(np.count_nonzero(spot_mask))
    lesion_area_ratio = lesion_pixels / leaf_area

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        spot_mask,
        connectivity=8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    if component_areas.size:
        component_ratios = component_areas / leaf_area
    else:
        component_ratios = np.array([], dtype=np.float64)

    if component_ratios.size == 0:
        avg_lesion_area = 0.0
        lesion_area_std = 0.0
    else:
        avg_lesion_area = float(np.mean(component_ratios))
        lesion_area_std = float(np.std(component_ratios))

    lesion_count = int(max(count - 1, 0))
    lesion_density = lesion_count / leaf_area
    largest_lesion_area_ratio = (
        float(np.max(component_ratios))
        if component_ratios.size
        else 0.0
    )

    lesions: list[LesionMeasurement] = []
    solidities: list[float] = []
    circularities: list[float] = []
    border_complexities: list[float] = []
    for label_idx in range(1, count):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = (labels == label_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))

        solidity = area / hull_area if hull_area > 0 else 0.0
        circularity = (
            (4.0 * math.pi * area) / (perimeter**2)
            if perimeter > 0
            else 0.0
        )
        border_complexity = (
            (perimeter**2) / area
            if area > 0
            else 0.0
        )

        lesions.append(LesionMeasurement(
            label_id=label_idx,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            solidity=solidity,
            centroid=_contour_centroid(contour),
        ))
        solidities.append(solidity)
        circularities.append(circularity)
        border_complexities.append(border_complexity)

    mean_lesion_solidity, std_lesion_solidity = _mean_std(solidities)
    mean_lesion_circularity, std_lesion_circularity = _mean_std(
        circularities
    )
    mean_border_complexity, std_border_complexity = _mean_std(
        border_complexities
    )

    summary = {
        "lesion_area_ratio": float(lesion_area_ratio),
        "lesion_count": lesion_count,
        "lesion_density": float(lesion_density),
        "avg_lesion_area": avg_lesion_area,
        "lesion_area_std": lesion_area_std,
        "largest_lesion_area_ratio": largest_lesion_area_ratio,
        "mean_lesion_solidity": mean_lesion_solidity,
        "std_lesion_solidity": std_lesion_solidity,
        "mean_lesion_circularity": mean_lesion_circularity,
        "std_lesion_circularity": std_lesion_circularity,
        "mean_border_complexity": mean_border_complexity,
        "std_border_complexity": std_border_complexity,
    }
    return tuple(lesions), summary


def compute_image_features(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    spot_mask: np.ndarray,
) -> ImageFeatures:
    leaf_features = _color_features_from_mask(image, leaf_mask)
    spot_features = _color_features_from_mask(image, spot_mask, prefix="spot_")

    leaf_ratios = {
        f"{name}_ratio": leaf_features[f"{name}_ratio"]
        for name in COLOR_NAMES
    }
    spot_ratios = {
        f"spot_{name}_ratio": spot_features[f"spot_{name}_ratio"]
        for name in COLOR_NAMES
    }
    leaf_stats = {
        key: leaf_features[key]
        for key in (
            "saturation_mean",
            "saturation_std",
            "value_mean",
            "value_std",
            "color_variance",
        )
    }
    spot_stats = {
        key: spot_features[f"spot_{key}"]
        for key in (
            "saturation_mean",
            "saturation_std",
            "value_mean",
            "value_std",
            "color_variance",
        )
    }

    lesions, lesion_summary = _measure_lesions(spot_mask, leaf_mask)
    return ImageFeatures(
        color=ImageColorFeatures(
            leaf_ratios=leaf_ratios,
            spot_ratios=spot_ratios,
            leaf_stats=leaf_stats,
            spot_stats=spot_stats,
        ),
        lesions=lesions,
        lesion_summary=lesion_summary,
    )


def image_features_to_row(
    image_path: Path,
    label: str,
    features: ImageFeatures,
) -> dict[str, object]:
    row: dict[str, object] = {
        "image_path": str(image_path),
        "label": label,
    }
    row.update(features.color.leaf_ratios)
    row.update(features.color.leaf_stats)
    row.update(features.color.spot_ratios)
    row.update(features.color.spot_stats)
    row.update(features.lesion_summary)
    return row
