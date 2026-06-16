from pathlib import Path
import random

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd
from tqdm import tqdm

from transformation.leaf_cache import get_leaf_mask
from transformation.mask import LeafMaskRecord
from transformation.spot_mask import (
    DEFAULT_SPOT_THRESHOLDS,
    _load_cached_spot_mask,
    _save_cached_spot_mask,
    build_spot_binary_mask,
)
from transformation.util import IMAGE_EXTENSIONS

SOURCE_DIR = Path("leaves/images")
OUTPUT_CSV = Path("outputs/classification/train/features_rf.csv")
IMAGES_PER_DIR = 200
RANDOM_SEED = 42

_COLOR_RATIO_KEYS = (
    "green_ratio",
    "yellow_ratio",
    "orange_ratio",
    "brown_ratio",
    "dark_ratio",
)


def _label_from_path(image_path: Path) -> str:
    if len(image_path.parts) >= 2:
        return image_path.parent.name
    return image_path.stem


def _sample_image_paths(
    src: Path,
    *,
    per_dir: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue

        selected.extend(rng.sample(images, min(per_dir, len(images))))

    return selected


def _empty_color_features(prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}{name}": 0.0
        for name in _COLOR_RATIO_KEYS
    } | {
        f"{prefix}saturation_mean": 0.0,
        f"{prefix}saturation_std": 0.0,
        f"{prefix}value_mean": 0.0,
        f"{prefix}value_std": 0.0,
        f"{prefix}color_variance": 0.0,
    }


def _color_features_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    masked_pixels = mask > 0
    pixel_count = int(np.count_nonzero(masked_pixels))
    if pixel_count == 0:
        return _empty_color_features(prefix)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    h = hue[masked_pixels]
    s = saturation[masked_pixels]
    v = value[masked_pixels]

    green = (
        (h >= 35)
        & (h <= 95)
        & (s >= 35)
        & (v >= 35)
    )
    yellow = (
        (h >= 20)
        & (h < 35)
        & (s >= 40)
        & (v >= 40)
    )
    orange = (
        (h >= 10)
        & (h < 20)
        & (s >= 45)
        & (v >= 35)
    )
    brown = (
        (h <= 25)
        & (s >= 40)
        & (v >= 15)
        & (v < 110)
    )
    dark = v < 45

    return {
        f"{prefix}green_ratio": float(np.count_nonzero(green) / pixel_count),
        f"{prefix}yellow_ratio": float(np.count_nonzero(yellow) / pixel_count),
        f"{prefix}orange_ratio": float(np.count_nonzero(orange) / pixel_count),
        f"{prefix}brown_ratio": float(np.count_nonzero(brown) / pixel_count),
        f"{prefix}dark_ratio": float(np.count_nonzero(dark) / pixel_count),
        f"{prefix}saturation_mean": float(np.mean(s)),
        f"{prefix}saturation_std": float(np.std(s)),
        f"{prefix}value_mean": float(np.mean(v)),
        f"{prefix}value_std": float(np.std(v)),
        f"{prefix}color_variance": float(np.var(h.astype(np.float64))),
    }


def _get_spot_mask(
    image_path: Path,
    image: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    spot_mask = _load_cached_spot_mask(image_path, DEFAULT_SPOT_THRESHOLDS)
    if spot_mask is not None:
        return spot_mask

    spot_mask = build_spot_binary_mask(
        image,
        leaf_mask,
        DEFAULT_SPOT_THRESHOLDS,
    )
    _save_cached_spot_mask(image_path, DEFAULT_SPOT_THRESHOLDS, spot_mask)
    return spot_mask


def _lesion_features(
    spot_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> dict[str, float | int]:
    leaf_area = max(int(np.count_nonzero(leaf_mask)), 1)
    lesion_pixels = int(np.count_nonzero(spot_mask))
    lesion_area_ratio = lesion_pixels / leaf_area

    count, _, stats, _ = cv2.connectedComponentsWithStats(
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

    return {
        "lesion_area_ratio": float(lesion_area_ratio),
        "lesion_count": int(max(count - 1, 0)),
        "avg_lesion_area": avg_lesion_area,
        "lesion_area_std": lesion_area_std,
    }


def _row_from_result(
    image_path: Path,
    record: LeafMaskRecord,
    image: np.ndarray,
) -> dict[str, object]:
    leaf_mask = record.mask
    spot_mask = _get_spot_mask(image_path, image, leaf_mask)

    row: dict[str, object] = {
        "image_path": str(image_path),
        "label": _label_from_path(image_path),
        "leaf_solidity": record.leaf_solidity,
        "mask_area_ratio": record.mask_area_ratio,
        "border_touch_ratio": record.border_touch_ratio,
    }
    row.update(_color_features_from_mask(image, leaf_mask))
    row.update(_color_features_from_mask(image, spot_mask, prefix="spot_"))
    row.update(_lesion_features(spot_mask, leaf_mask))
    return row


def extract_features(image_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for image_path in tqdm(
        image_paths,
        desc="extract",
        unit="image",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        record = get_leaf_mask(image_path)
        if record is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        rows.append(_row_from_result(image_path, record, image))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract(
    *,
    src: Path = SOURCE_DIR,
    per_dir: int = IMAGES_PER_DIR,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    image_paths = _sample_image_paths(
        src,
        per_dir=per_dir,
        seed=seed,
    )
    return extract_features(image_paths)


def save_features_csv(
    df: pd.DataFrame,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    df = extract()
    csv_path = save_features_csv(df)
    print(f"Extracted: {len(df)} leaves")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
