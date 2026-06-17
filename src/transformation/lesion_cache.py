"""Persistent spot mask and lesion-feature cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from transformation.cache_util import (
    cache_path_for,
    decode_mask,
    encode_mask,
    load_json_cache,
    save_json_cache,
)
from transformation.image_features import (
    ImageColorFeatures,
    ImageFeatures,
    compute_image_features,
)
from transformation.leaf_cache import get_leaf_mask
from transformation.spot_mask import (
    DEFAULT_SPOT_THRESHOLDS,
    SpotThresholds,
    build_spot_binary_mask,
    thresholds_signature,
)

CACHE_ROOT = Path("outputs/cache/lesion")


@dataclass(frozen=True)
class LesionRecord:
    spot_mask: np.ndarray
    color: ImageColorFeatures
    lesion_summary: dict[str, float | int]


def _serialize_color(color: ImageColorFeatures) -> dict[str, object]:
    return {
        "leaf_ratios": color.leaf_ratios,
        "spot_ratios": color.spot_ratios,
        "leaf_stats": color.leaf_stats,
        "spot_stats": color.spot_stats,
    }


def _deserialize_color(
    payload: dict[str, object],
) -> ImageColorFeatures | None:
    try:
        return ImageColorFeatures(
            leaf_ratios=dict(payload["leaf_ratios"]),  # type: ignore[arg-type]
            spot_ratios=dict(payload["spot_ratios"]),  # type: ignore[arg-type]
            leaf_stats=dict(payload["leaf_stats"]),  # type: ignore[arg-type]
            spot_stats=dict(payload["spot_stats"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_cached_record(
    image_path: Path,
    thresholds: SpotThresholds,
) -> LesionRecord | None:
    cache_path = cache_path_for(image_path, CACHE_ROOT)
    payload = load_json_cache(cache_path)
    if payload is None:
        return None

    if payload.get("source_mtime") != image_path.stat().st_mtime:
        return None
    if payload.get("thresholds_signature") != thresholds_signature(thresholds):
        return None

    spot_payload = payload.get("spot_mask", {})
    if not isinstance(spot_payload, dict):
        return None
    spot_mask = decode_mask(spot_payload)
    color = _deserialize_color(payload.get("color", {}))
    summary = payload.get("lesion_summary")
    if spot_mask is None or color is None or not isinstance(summary, dict):
        return None

    return LesionRecord(
        spot_mask=spot_mask,
        color=color,
        lesion_summary=summary,
    )


def _save_cached_record(
    image_path: Path,
    record: LesionRecord,
    thresholds: SpotThresholds,
) -> None:
    cache_path = cache_path_for(image_path, CACHE_ROOT)
    payload = {
        "source_path": str(image_path),
        "source_mtime": image_path.stat().st_mtime,
        "thresholds_signature": thresholds_signature(thresholds),
        "spot_mask": encode_mask(record.spot_mask),
        "color": _serialize_color(record.color),
        "lesion_summary": record.lesion_summary,
    }
    save_json_cache(cache_path, payload)


def _record_from_features(
    spot_mask: np.ndarray,
    features: ImageFeatures,
) -> LesionRecord:
    return LesionRecord(
        spot_mask=spot_mask,
        color=features.color,
        lesion_summary=features.lesion_summary,
    )


def compute_lesion_record(
    image: np.ndarray,
    image_path: Path,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
) -> LesionRecord:
    spot_mask = build_spot_binary_mask(image, leaf_mask, thresholds)
    features = compute_image_features(image, leaf_mask, spot_mask)
    record = _record_from_features(spot_mask, features)
    _save_cached_record(image_path, record, thresholds)
    return record


def get_lesion_record(
    image_path: Path,
    image: np.ndarray,
    leaf_mask: np.ndarray,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
    *,
    force: bool = False,
) -> LesionRecord:
    if not force:
        cached = _load_cached_record(image_path, thresholds)
        if cached is not None:
            return cached
    return compute_lesion_record(image, image_path, leaf_mask, thresholds)


def ensure_lesion_record(
    image_path: Path,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
    *,
    image: np.ndarray | None = None,
    force: bool = False,
) -> LesionRecord | None:
    """Load or compute the lesion cache without writing any PNG output."""
    image_path = Path(image_path)
    if not image_path.is_file():
        return None

    if not force:
        cached = _load_cached_record(image_path, thresholds)
        if cached is not None:
            return cached

    leaf_record = get_leaf_mask(image_path, image=image, force=force)
    if leaf_record is None:
        return None

    if image is None:
        image = cv2.imread(str(image_path))
    if image is None:
        return None

    return compute_lesion_record(
        image,
        image_path,
        leaf_record.mask,
        thresholds,
    )


def get_spot_mask(
    image_path: Path,
    thresholds: SpotThresholds = DEFAULT_SPOT_THRESHOLDS,
    *,
    force: bool = False,
) -> np.ndarray | None:
    record = ensure_lesion_record(image_path, thresholds, force=force)
    if record is None:
        return None
    return record.spot_mask


def lesion_record_to_image_features(record: LesionRecord) -> ImageFeatures:
    return ImageFeatures(
        color=record.color,
        lesions=(),
        lesion_summary=record.lesion_summary,
    )
