"""Shared helpers for mask JSON caches.

Terminology:
- leaf_mask: leaf segmentation mask
- spot_mask: binary diseased-region mask
- lesion: connected component inside spot_mask
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]


def cache_path_for(image_path: Path, cache_root: Path) -> Path:
    return cache_root / image_path.parent.name / f"{image_path.stem}.json"


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    packed = np.packbits((mask > 0).astype(np.uint8))
    return {
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "packed_bits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def decode_mask(payload: dict[str, object]) -> np.ndarray | None:
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


def load_json_cache(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def save_json_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def relative_source_path(image_path: Path) -> Path:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(Path.cwd())
    except ValueError:
        return resolved
