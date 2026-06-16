import base64
import json
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from transformation.mask import LeafMaskRecord, compute_leaf_mask_record

CACHE_ROOT = Path("outputs/cache/leaf_mask")
LEGACY_CACHE_ROOT = Path("outputs/cache/leaf_masks")
DEFAULT_DATASET_ROOT = Path("leaves/images")


def _relative_source_path(image_path: Path) -> Path:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(Path.cwd())
    except ValueError:
        return resolved


def _image_cache_key(image_path: Path) -> Path:
    # cache/leaf_mask/<class>/<image_stem>.json
    return Path(image_path.parent.name) / f"{image_path.stem}.json"


def _cache_path(image_path: Path) -> Path:
    return CACHE_ROOT / _image_cache_key(image_path)


def _encode_mask(mask: np.ndarray) -> dict[str, object]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
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
    mask = unpacked[:pixel_count].reshape(height, width).astype(np.uint8) * 255
    return mask


def _load_cached_record(
    image_path: Path,
    cache_path: Path,
) -> LeafMaskRecord | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    source_mtime = image_path.stat().st_mtime
    if meta.get("source_mtime") != source_mtime:
        return None
    mask = _decode_mask(meta.get("mask", {}))
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": _relative_source_path(image_path).as_posix(),
        "source_mtime": image_path.stat().st_mtime,
        "leaf_solidity": record.leaf_solidity,
        "mask_area_ratio": record.mask_area_ratio,
        "border_touch_ratio": record.border_touch_ratio,
        "mask": _encode_mask(record.mask),
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _migrate_legacy_cache(
    image_path: Path,
    cache_path: Path,
) -> LeafMaskRecord | None:
    rel = _relative_source_path(image_path)
    legacy_png = LEGACY_CACHE_ROOT / rel.with_suffix(".png")
    legacy_json = LEGACY_CACHE_ROOT / rel.with_suffix(".json")
    if not legacy_png.is_file() or not legacy_json.is_file():
        return None
    try:
        with legacy_json.open(encoding="utf-8") as handle:
            legacy_meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
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
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> LeafMaskRecord | None:
    del dataset_root
    image_path = Path(image_path)
    if not image_path.is_file():
        return None
    cache_path = _cache_path(image_path)
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
