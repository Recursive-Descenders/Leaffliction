"""Persistent leaf mask cache."""

from pathlib import Path

import cv2  # type: ignore[import-not-found]

from transformation.cache_util import (
    cache_path_for,
    decode_mask,
    encode_mask,
    load_json_cache,
    relative_source_path,
    save_json_cache,
)
from transformation.mask import LeafMaskRecord, compute_leaf_mask_record

CACHE_ROOT = Path("outputs/cache/leaf_mask")
LEGACY_CACHE_ROOT = Path("outputs/cache/leaf_masks")


def _load_cached_record(
    image_path: Path,
    cache_path: Path,
) -> LeafMaskRecord | None:
    meta = load_json_cache(cache_path)
    if meta is None:
        return None
    if meta.get("source_mtime") != image_path.stat().st_mtime:
        return None
    mask_payload = meta.get("mask", {})
    if not isinstance(mask_payload, dict):
        return None
    mask = decode_mask(mask_payload)
    if mask is None:
        return None
    return LeafMaskRecord(
        mask=mask,
        leaf_solidity=float(meta["leaf_solidity"]),
        mask_area_ratio=float(meta["mask_area_ratio"]),
        border_touch_ratio=float(meta["border_touch_ratio"]),
    )


def _save_cached_record(
    image_path: Path,
    record: LeafMaskRecord,
    cache_path: Path,
) -> None:
    payload = {
        "source_path": relative_source_path(image_path).as_posix(),
        "source_mtime": image_path.stat().st_mtime,
        "leaf_solidity": record.leaf_solidity,
        "mask_area_ratio": record.mask_area_ratio,
        "border_touch_ratio": record.border_touch_ratio,
        "mask": encode_mask(record.mask),
    }
    save_json_cache(cache_path, payload)


def _migrate_legacy_cache(
    image_path: Path,
    cache_path: Path,
) -> LeafMaskRecord | None:
    rel = relative_source_path(image_path)
    legacy_png = LEGACY_CACHE_ROOT / rel.with_suffix(".png")
    legacy_json = LEGACY_CACHE_ROOT / rel.with_suffix(".json")
    if not legacy_png.is_file() or not legacy_json.is_file():
        return None
    legacy_meta = load_json_cache(legacy_json)
    if legacy_meta is None:
        return None
    mask = cv2.imread(str(legacy_png), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    record = LeafMaskRecord(
        mask=mask,
        leaf_solidity=float(legacy_meta["leaf_solidity"]),
        mask_area_ratio=float(legacy_meta["mask_area_ratio"]),
        border_touch_ratio=float(legacy_meta["border_touch_ratio"]),
    )
    _save_cached_record(image_path, record, cache_path)
    try:
        legacy_png.unlink(missing_ok=True)
        legacy_json.unlink(missing_ok=True)
    except OSError:
        pass
    return record


def get_leaf_mask(
    image_path: Path,
    *,
    force: bool = False,
) -> LeafMaskRecord | None:
    image_path = Path(image_path)
    if not image_path.is_file():
        return None

    cache_path = cache_path_for(image_path, CACHE_ROOT)
    if not force:
        cached = _load_cached_record(image_path, cache_path)
        if cached is not None:
            return cached
        migrated = _migrate_legacy_cache(image_path, cache_path)
        if migrated is not None:
            return migrated

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    record = compute_leaf_mask_record(image)
    if record is None:
        return None
    _save_cached_record(image_path, record, cache_path)
    return record
