"""Spot mask generation and lesion contour drawing.

Terminology:
- spot_mask: binary diseased-region mask inside the leaf
- lesion: connected component inside spot_mask
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

SPOT_CONTOUR_COLOR = (0, 0, 255)
CONTOUR_THICKNESS = 2

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class SpotThresholds:
    shadow_v_max: int
    shadow_s_max: int
    over_v_min: int
    over_s_max: int
    spot_h_ranges: tuple[tuple[int, int], ...]
    spot_s_min: int
    spot_v_min: int
    spot_v_max: int
    min_spot_area_ratio: float
    max_spot_area_ratio: float


DEFAULT_SPOT_THRESHOLDS = SpotThresholds(
    shadow_v_max=25,
    shadow_s_max=46,
    over_v_min=221,
    over_s_max=39,
    spot_h_ranges=((0, 29), (160, 179)),
    spot_s_min=26,
    spot_v_min=26,
    spot_v_max=199,
    min_spot_area_ratio=0.0006,
    max_spot_area_ratio=0.0323,
)


def _hue_in_ranges(
    hue: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    result = np.zeros(hue.shape, dtype=bool)
    for low, high in ranges:
        result |= (hue >= low) & (hue <= high)
    return result


def thresholds_signature(thresholds: SpotThresholds) -> str:
    raw = (
        f"{thresholds.shadow_v_max}|{thresholds.shadow_s_max}|"
        f"{thresholds.over_v_min}|{thresholds.over_s_max}|"
        f"{thresholds.spot_h_ranges}|{thresholds.spot_s_min}|"
        f"{thresholds.spot_v_min}|{thresholds.spot_v_max}|"
        f"{thresholds.min_spot_area_ratio:.6f}|"
        f"{thresholds.max_spot_area_ratio:.6f}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_excluded(
    s: np.ndarray,
    v: np.ndarray,
    thresholds: SpotThresholds,
) -> np.ndarray:
    # Keep this shadow definition aligned with mask.py refinement:
    # low-value + low-saturation regions are likely illumination shadow.
    shadow = (
        (v <= thresholds.shadow_v_max)
        & (s <= thresholds.shadow_s_max)
    )
    over = (
        (v >= thresholds.over_v_min)
        & (s <= thresholds.over_s_max)
    )
    return shadow | over


def _is_spot_colored(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    thresholds: SpotThresholds,
) -> np.ndarray:
    return (
        _hue_in_ranges(h, thresholds.spot_h_ranges)
        & (s >= thresholds.spot_s_min)
        & (v >= thresholds.spot_v_min)
        & (v <= thresholds.spot_v_max)
    )


def _filter_blobs_by_area(
    candidate_mask: np.ndarray,
    leaf_mask: np.ndarray,
    min_ratio: float,
    max_ratio: float,
) -> np.ndarray:
    leaf_area = max(int(np.count_nonzero(leaf_mask)), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    filtered = np.zeros_like(candidate_mask)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        ratio = area / leaf_area
        if min_ratio <= ratio <= max_ratio:
            filtered[labels == label] = 255

    return filtered


def build_spot_binary_mask(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    in_leaf = leaf_mask > 0
    excluded = _is_excluded(s_channel, v_channel, thresholds)
    spot = (
        in_leaf
        & ~excluded
        & _is_spot_colored(
            h_channel,
            s_channel,
            v_channel,
            thresholds,
        )
    )

    candidate = np.zeros_like(leaf_mask)
    candidate[spot] = 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, _OPEN_KERNEL)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    candidate &= leaf_mask

    return _filter_blobs_by_area(
        candidate,
        leaf_mask,
        thresholds.min_spot_area_ratio,
        thresholds.max_spot_area_ratio,
    )


def _spot_contours(
    spot_mask: np.ndarray,
    leaf_mask: np.ndarray,
    min_ratio: float,
) -> list[np.ndarray]:
    leaf_area = max(int(np.count_nonzero(leaf_mask)), 1)
    min_area = leaf_area * min_ratio

    contours, _ = cv2.findContours(
        spot_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]


def draw_lesion_contours(
    image: np.ndarray,
    spot_mask: np.ndarray,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> np.ndarray:
    contours = _spot_contours(
        spot_mask,
        leaf_mask,
        thresholds.min_spot_area_ratio,
    )
    overlay = image.copy()
    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=SPOT_CONTOUR_COLOR,
        thickness=CONTOUR_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return overlay
