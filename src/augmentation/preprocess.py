"""
Pre-processing utilities for the dataset-level augmentor (mode C).

Currently exposes :func:`dedup_images`, which drops byte-identical duplicates
from a list of image paths (decision #6 — exact file-hash). Order of the
result follows the input order; the first occurrence of each hash wins.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(folder: Path) -> list[Path]:
    """Return image files directly inside ``folder`` (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def dedup_images(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Drop byte-identical duplicates.

    Returns ``(kept, dropped)`` so callers can log what was removed.
    """
    seen: dict[str, Path] = {}
    kept: list[Path] = []
    dropped: list[Path] = []
    for path in paths:
        h = file_hash(path)
        if h in seen:
            dropped.append(path)
        else:
            seen[h] = path
            kept.append(path)
    return kept, dropped
