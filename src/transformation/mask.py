from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from transformation.parallel import parallel_map
from transformation.util import (
    get_image_paths,
    largest_leaf_mask,
    make_output_path,
    skip_image,
    validate_image_readable,
)

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_REFINED_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 2
EDGE_MARGIN = 3
SHADOW_V_MAX = 30
SHADOW_S_MAX = 45


@dataclass(frozen=True)
class LeafMaskRecord:
    mask: np.ndarray
    leaf_solidity: float
    mask_area_ratio: float
    border_touch_ratio: float


def _hsv_leaf_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    not_black = v_channel > 25
    colored_leaf = s_channel > 25
    hue_leaf_range = h_channel <= 95
    gray_background = (
        (s_channel < 45)
        & (v_channel > 35)
        & (v_channel < 190)
    )

    mask_bool = (
        not_black
        & colored_leaf
        & hue_leaf_range
        & ~gray_background
    )
    return mask_bool.astype(np.uint8) * 255


def build_mask(image: np.ndarray) -> np.ndarray:
    mask_image = _hsv_leaf_mask(image)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return mask_image


def _shadow_like_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    return (
        (v_channel <= SHADOW_V_MAX)
        & (s_channel <= SHADOW_S_MAX)
    )


def build_refined_mask(image: np.ndarray) -> np.ndarray:
    """HSV seed + GrabCut for centered leaves on gray background."""
    hsv_mask = build_mask(image)
    height, width = image.shape[:2]
    margin_x = int(width * 0.06)
    margin_y = int(height * 0.06)

    if width - 2 * margin_x <= 1 or height - 2 * margin_y <= 1:
        return hsv_mask

    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

    grab_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    inner = grab_mask[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ]
    inner[:] = cv2.GC_PR_FGD
    inner_hsv = hsv_mask[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ]
    inner[inner_hsv > 0] = cv2.GC_PR_FGD
    shadow_like = _shadow_like_mask(image)
    inner_shadow = shadow_like[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ]
    inner[inner_shadow] = cv2.GC_PR_BGD

    center_y, center_x = height // 2, width // 2
    seed = 25
    grab_mask[
        max(0, center_y - seed):min(height, center_y + seed),
        max(0, center_x - seed):min(width, center_x + seed),
    ] = cv2.GC_FGD

    has_background = np.any(grab_mask == cv2.GC_BGD)
    has_foreground = np.any(
        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)
    )
    if not has_background or not has_foreground:
        return hsv_mask

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            image,
            grab_mask,
            rect,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return hsv_mask

    refined = np.where(
        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    refined[shadow_like] = 0
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, _OPEN_KERNEL)
    return cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _REFINED_CLOSE_KERNEL)


def _largest_contour(leaf_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def _near_border(
    x: float,
    y: float,
    width: int,
    height: int,
) -> bool:
    return (
        x <= EDGE_MARGIN
        or x >= width - 1 - EDGE_MARGIN
        or y <= EDGE_MARGIN
        or y >= height - 1 - EDGE_MARGIN
    )


def _border_touch_ratio(leaf_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).astype(float)
    if len(points) < 2:
        return 0.0

    height, width = leaf_mask.shape
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0

    border_length = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        segment = float(np.hypot(x2 - x1, y2 - y1))
        on_p1 = _near_border(x1, y1, width, height)
        on_p2 = _near_border(x2, y2, width, height)
        if on_p1 and on_p2:
            border_length += segment
        elif on_p1 or on_p2:
            border_length += segment * 0.5

    return border_length / perimeter


def _mask_quality_metrics(
    leaf_mask: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, float, float]:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    leaf_solidity = area / hull_area if hull_area else 0.0
    image_area = leaf_mask.shape[0] * leaf_mask.shape[1]
    mask_area_ratio = np.count_nonzero(leaf_mask) / image_area
    touch_ratio = _border_touch_ratio(leaf_mask)

    return leaf_solidity, mask_area_ratio, touch_ratio


def _draw_contour_overlay(
    image: np.ndarray,
    contour: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    cv2.drawContours(
        overlay,
        [contour],
        contourIdx=0,
        color=color,
        thickness=CONTOUR_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return overlay


def compute_leaf_mask_record(image: np.ndarray) -> LeafMaskRecord | None:
    leaf_mask = largest_leaf_mask(build_refined_mask(image))
    if leaf_mask is None:
        return None

    contour = _largest_contour(leaf_mask)
    if contour is None:
        return None

    leaf_solidity, mask_area_ratio, touch_ratio = _mask_quality_metrics(
        leaf_mask,
        contour,
    )
    return LeafMaskRecord(
        mask=leaf_mask,
        leaf_solidity=leaf_solidity,
        mask_area_ratio=mask_area_ratio,
        border_touch_ratio=touch_ratio,
    )


def _overlay_from_record(
    image: np.ndarray,
    record: LeafMaskRecord,
) -> np.ndarray | None:
    contour = _largest_contour(record.mask)
    if contour is None:
        return None
    return _draw_contour_overlay(image, contour, CONTOUR_COLOR)


def apply_mask(
    image: np.ndarray,
    image_path: Path | None = None,
) -> np.ndarray | None:
    if image_path is not None:
        from transformation.leaf_cache import get_leaf_mask

        record = get_leaf_mask(image_path)
    else:
        record = compute_leaf_mask_record(image)

    if record is None:
        return None
    return _overlay_from_record(image, record)


@dataclass(frozen=True)
class MaskWorkItem:
    src: str
    dst: str
    file: str | None
    image_path: str


def process_mask_image(item: MaskWorkItem) -> str | None:
    src = Path(item.src)
    dst = Path(item.dst)
    image_path = Path(item.image_path)
    file = item.file

    try:
        image = validate_image_readable(image_path)
    except ValueError as exc:
        skip_image(image_path, str(exc))
        return None

    from transformation.leaf_cache import get_leaf_mask

    record = get_leaf_mask(image_path, image=image)
    if record is None:
        skip_image(image_path, "no detected leaf")
        return None

    overlay = _overlay_from_record(image, record)
    if overlay is None:
        skip_image(image_path, "no detected leaf")
        return None

    output_file = make_output_path(src, dst, image_path, file, "mask")
    cv2.imwrite(str(output_file), overlay)
    return str(output_file)


def mask(
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
        MaskWorkItem(
            src=str(src),
            dst=str(dst),
            file=file_name,
            image_path=str(image_path),
        )
        for image_path in get_image_paths(src, file)
    ]
    results = parallel_map(
        process_mask_image,
        items,
        jobs=jobs,
        desc="mask",
    )
    return [Path(path) for path in results if path is not None]
