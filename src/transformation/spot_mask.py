import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from transformation.leaf_cache import get_leaf_mask
from transformation.mask import compute_leaf_mask_record
from transformation.util import (
    IMAGE_EXTENSIONS,
    iter_image_paths,
    make_output_path,
)

SPOT_CONTOUR_COLOR = (0, 0, 255)
CONTOUR_THICKNESS = 2
SPOT_CACHE_ROOT = Path("outputs/cache/spot_mask")

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

HEALTHY_CLASSES = frozenset({"Apple_healthy", "Grape_healthy"})
DISEASED_CLASSES = frozenset({
    "Apple_Black_rot",
    "Apple_rust",
    "Apple_scab",
    "Grape_Black_rot",
    "Grape_Esca",
    "Grape_spot",
})

ROUGH_SHADOW_V = 25
ROUGH_SHADOW_S = 50
ROUGH_OVER_V = 220
ROUGH_OVER_S = 40
SAMPLE_GREEN_H_MIN = 28
SAMPLE_GREEN_H_MAX = 90
SAMPLE_GREEN_S_MIN = 26
SAMPLE_PROVISIONAL_MIN_AREA = 0.0005
SAMPLE_PROVISIONAL_MAX_AREA = 0.40
MIN_AREA_RATIO_FLOOR = 0.0005


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


# Derived from sample_spot_thresholds() on leaves/images (all images/class).
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


def _spot_cache_path(image_path: Path) -> Path:
    return (
        SPOT_CACHE_ROOT
        / image_path.parent.name
        / f"{image_path.stem}.json"
    )


def _thresholds_signature(thresholds: SpotThresholds) -> str:
    raw = (
        f"{thresholds.shadow_v_max}|{thresholds.shadow_s_max}|"
        f"{thresholds.over_v_min}|{thresholds.over_s_max}|"
        f"{thresholds.spot_h_ranges}|{thresholds.spot_s_min}|"
        f"{thresholds.spot_v_min}|{thresholds.spot_v_max}|"
        f"{thresholds.min_spot_area_ratio:.6f}|"
        f"{thresholds.max_spot_area_ratio:.6f}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _encode_mask(mask: np.ndarray) -> dict[str, object]:
    packed = np.packbits((mask > 0).astype(np.uint8))
    return {
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "packed_bits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _decode_mask(payload: dict[str, object]) -> np.ndarray | None:
    shape = payload.get("shape")
    packed_b64 = payload.get("packed_bits_b64")
    if not isinstance(shape, list) or len(shape) != 2:
        return None
    if not isinstance(packed_b64, str):
        return None
    height, width = int(shape[0]), int(shape[1])
    pixel_count = height * width
    try:
        packed = np.frombuffer(base64.b64decode(packed_b64), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    unpacked = np.unpackbits(packed)
    if unpacked.size < pixel_count:
        return None
    return unpacked[:pixel_count].reshape(height, width).astype(np.uint8) * 255


def _load_cached_spot_mask(
    image_path: Path,
    thresholds: SpotThresholds,
) -> np.ndarray | None:
    cache_path = _spot_cache_path(image_path)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("source_mtime") != image_path.stat().st_mtime:
        return None
    signature = _thresholds_signature(thresholds)
    if payload.get("thresholds_signature") != signature:
        return None
    mask = _decode_mask(payload.get("spot_mask", {}))
    return mask


def _save_cached_spot_mask(
    image_path: Path,
    thresholds: SpotThresholds,
    spot_mask: np.ndarray,
) -> None:
    cache_path = _spot_cache_path(image_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(image_path),
        "source_mtime": image_path.stat().st_mtime,
        "thresholds_signature": _thresholds_signature(thresholds),
        "spot_mask": _encode_mask(spot_mask),
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def _draw_contour_overlay(
    image: np.ndarray,
    contours: list[np.ndarray],
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=color,
        thickness=CONTOUR_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return overlay


def apply_spot_mask(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> np.ndarray:
    spot_mask = build_spot_binary_mask(image, leaf_mask, thresholds)
    contours = _spot_contours(
        spot_mask,
        leaf_mask,
        thresholds.min_spot_area_ratio,
    )
    return _draw_contour_overlay(image, contours, SPOT_CONTOUR_COLOR)


def apply_spot_mask_from_path(
    image_path: Path,
    image: np.ndarray,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> np.ndarray:
    spot_mask = _load_cached_spot_mask(image_path, thresholds)
    if spot_mask is None:
        spot_mask = build_spot_binary_mask(image, leaf_mask, thresholds)
        _save_cached_spot_mask(image_path, thresholds, spot_mask)
    contours = _spot_contours(
        spot_mask,
        leaf_mask,
        thresholds.min_spot_area_ratio,
    )
    return _draw_contour_overlay(image, contours, SPOT_CONTOUR_COLOR)


def _class_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _rough_spot_candidate_mask(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    shadow_v_max: int,
    shadow_s_max: int,
    over_v_min: int,
    over_s_max: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    in_leaf = leaf_mask > 0
    shadow = (v_channel <= shadow_v_max) & (s_channel <= shadow_s_max)
    over = (v_channel >= over_v_min) & (s_channel <= over_s_max)
    colored = in_leaf & ~shadow & ~over

    green_guess = (
        (h_channel >= SAMPLE_GREEN_H_MIN)
        & (h_channel <= SAMPLE_GREEN_H_MAX)
        & (s_channel >= SAMPLE_GREEN_S_MIN)
    )
    candidates = colored & ~green_guess & (s_channel >= 20)

    mask = np.zeros_like(leaf_mask)
    mask[candidates] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    mask &= leaf_mask

    return _filter_blobs_by_area(
        mask,
        leaf_mask,
        SAMPLE_PROVISIONAL_MIN_AREA,
        SAMPLE_PROVISIONAL_MAX_AREA,
    )


def _blob_area_ratios(
    blob_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    leaf_area = max(int(np.count_nonzero(leaf_mask)), 1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        blob_mask,
        connectivity=8,
    )
    ratios: list[float] = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        ratios.append(area / leaf_area)
    return np.array(ratios, dtype=np.float64)


def _percentile_bounds(
    values: np.ndarray,
    low: float = 5.0,
    high: float = 95.0,
) -> tuple[int, int]:
    if values.size == 0:
        return 0, 0
    p_low, p_high = np.percentile(values, [low, high])
    return int(p_low), int(p_high)


def _merge_hue_ranges(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    if values.size == 0:
        return ()

    ranges: list[tuple[int, int]] = []
    for low, high in ((0, 45), (160, 179)):
        subset = values[(values >= low) & (values <= high)]
        if subset.size == 0:
            continue
        h_min, h_max = _percentile_bounds(subset, 2.0, 98.0)
        ranges.append((h_min, h_max))

    return tuple(ranges)


def _union_hue_ranges(
    ranges_list: list[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for ranges in ranges_list:
        merged.extend(ranges)

    if not merged:
        return ((0, 14), (10, 36), (165, 179))

    merged.sort(key=lambda item: item[0])
    combined: list[tuple[int, int]] = [merged[0]]
    for low, high in merged[1:]:
        prev_low, prev_high = combined[-1]
        if low <= prev_high + 2:
            combined[-1] = (prev_low, max(prev_high, high))
        else:
            combined.append((low, high))

    return tuple(combined)


def sample_spot_thresholds(
    src: str | Path = Path("leaves/images"),
) -> SpotThresholds:
    src = Path(src)

    all_shadow_s: list[np.ndarray] = []
    all_shadow_v: list[np.ndarray] = []
    all_over_s: list[np.ndarray] = []
    all_over_v: list[np.ndarray] = []
    per_class_spot_h: list[np.ndarray] = []
    per_class_spot_s: list[np.ndarray] = []
    all_spot_v: list[np.ndarray] = []
    per_class_area_ratios: list[np.ndarray] = []

    for class_name in sorted(DISEASED_CLASSES):
        class_dir = src / class_name
        if not class_dir.is_dir():
            print(f"Skipped missing class directory: {class_dir}")
            continue

        image_paths = _class_image_paths(class_dir)
        class_spot_h: list[np.ndarray] = []
        class_spot_s: list[np.ndarray] = []
        class_spot_v: list[np.ndarray] = []
        class_area_ratios: list[float] = []

        for image_path in image_paths:
            record = get_leaf_mask(image_path)
            if record is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            leaf_mask = record.mask
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            in_leaf = leaf_mask > 0
            shadow = (
                (v_channel <= ROUGH_SHADOW_V)
                & (s_channel <= ROUGH_SHADOW_S)
            )
            over = (
                (v_channel >= ROUGH_OVER_V)
                & (s_channel <= ROUGH_OVER_S)
            )
            if np.any(shadow):
                all_shadow_s.append(s_channel[in_leaf & shadow])
                all_shadow_v.append(v_channel[in_leaf & shadow])
            if np.any(over):
                all_over_s.append(s_channel[in_leaf & over])
                all_over_v.append(v_channel[in_leaf & over])

            blob_mask = _rough_spot_candidate_mask(
                image,
                leaf_mask,
                ROUGH_SHADOW_V,
                ROUGH_SHADOW_S,
                ROUGH_OVER_V,
                ROUGH_OVER_S,
            )
            if not np.any(blob_mask):
                continue

            class_spot_h.append(h_channel[blob_mask > 0])
            class_spot_s.append(s_channel[blob_mask > 0])
            class_spot_v.append(v_channel[blob_mask > 0])
            class_area_ratios.extend(
                _blob_area_ratios(blob_mask, leaf_mask).tolist()
            )

        if class_spot_h:
            per_class_spot_h.append(np.concatenate(class_spot_h))
            per_class_spot_s.append(np.concatenate(class_spot_s))
            all_spot_v.append(np.concatenate(class_spot_v))
        if class_area_ratios:
            per_class_area_ratios.append(
                np.array(class_area_ratios, dtype=np.float64)
            )

        area_summary = (
            f"p95={np.percentile(class_area_ratios, 95):.4f}"
            if class_area_ratios
            else "no blobs"
        )
        print(
            f"{class_name}: images={len(image_paths)} "
            f"spot_pixels={sum(arr.size for arr in class_spot_h)} "
            f"area {area_summary}"
        )

    shadow_v = (
        int(np.percentile(np.concatenate(all_shadow_v), 95))
        if all_shadow_v
        else ROUGH_SHADOW_V
    )
    shadow_s = (
        int(np.percentile(np.concatenate(all_shadow_s), 95))
        if all_shadow_s
        else ROUGH_SHADOW_S
    )
    over_s_max = (
        int(np.percentile(np.concatenate(all_over_s), 95))
        if all_over_s
        else ROUGH_OVER_S
    )
    over_v_min = (
        int(np.percentile(np.concatenate(all_over_v), 5))
        if all_over_v
        else ROUGH_OVER_V
    )

    per_class_h_ranges = [
        _merge_hue_ranges(class_h)
        for class_h in per_class_spot_h
    ]
    spot_h_ranges = _union_hue_ranges(per_class_h_ranges)

    spot_s_values = [
        int(np.percentile(class_s, 5))
        for class_s in per_class_spot_s
        if class_s.size
    ]
    spot_s_min = min(spot_s_values) if spot_s_values else 24

    spot_v = (
        np.concatenate(all_spot_v)
        if all_spot_v
        else np.array([26, 205])
    )
    spot_v_min, spot_v_max = _percentile_bounds(spot_v, 2.0, 98.0)

    all_ratios = (
        np.concatenate(per_class_area_ratios)
        if per_class_area_ratios
        else np.array([0.001, 0.15])
    )
    min_spot_area_ratio = max(
        MIN_AREA_RATIO_FLOOR,
        float(np.percentile(all_ratios, 5)),
    )
    per_class_p95 = [
        float(np.percentile(ratios, 95))
        for ratios in per_class_area_ratios
        if ratios.size
    ]
    max_spot_area_ratio = max(per_class_p95) if per_class_p95 else 0.15

    thresholds = SpotThresholds(
        shadow_v_max=shadow_v,
        shadow_s_max=shadow_s,
        over_v_min=over_v_min,
        over_s_max=over_s_max,
        spot_h_ranges=spot_h_ranges,
        spot_s_min=spot_s_min,
        spot_v_min=max(spot_v_min, shadow_v + 1),
        spot_v_max=min(spot_v_max, over_v_min - 1),
        min_spot_area_ratio=min_spot_area_ratio,
        max_spot_area_ratio=max_spot_area_ratio,
    )

    print("Derived spot thresholds:")
    print(
        f"  shadow_v_max={thresholds.shadow_v_max}, "
        f"shadow_s_max={thresholds.shadow_s_max}"
    )
    print(
        f"  over_v_min={thresholds.over_v_min}, "
        f"over_s_max={thresholds.over_s_max}"
    )
    print(f"  spot_h_ranges={thresholds.spot_h_ranges}")
    print(
        f"  spot_s_min={thresholds.spot_s_min}, "
        f"spot_v=[{thresholds.spot_v_min}, {thresholds.spot_v_max}]"
    )
    print(
        f"  spot_area_ratio=[{thresholds.min_spot_area_ratio:.4f}, "
        f"{thresholds.max_spot_area_ratio:.4f}]"
    )
    return thresholds


def apply_spot_mask_image(
    image: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> np.ndarray | None:
    record = compute_leaf_mask_record(image)
    if record is None:
        return None
    return apply_spot_mask(image, record.mask, thresholds)


def spot_mask(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="spot_mask"):
        if image_path.parent.name in HEALTHY_CLASSES:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        record = get_leaf_mask(image_path)
        if record is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        result = apply_spot_mask_from_path(image_path, image, record.mask)
        output_file = make_output_path(
            src, dst, image_path, file, "spot_mask"
        )
        cv2.imwrite(str(output_file), result)
        saved_paths.append(output_file)

    return saved_paths


if __name__ == "__main__":
    sample_spot_thresholds()
