"""
IO plumbing for Augmentation (load / name / save).
"""
from pathlib import Path
import cv2
import numpy as np


def make_test_image(size: int = 400) -> np.ndarray:
    """
    Synthetic image for visualizing augmentations without uploading anything.
    Asymmetric leaf-like pattern with corner marker so orientation is obvious.
    """
    img = np.full((size, size, 3), 248, dtype=np.uint8)
    cell = size // 8

    for j in range(8):
        for i in range(8):
            if (i + j) % 2 == 0:
                img[j*cell:(j+1)*cell, i*cell:(i+1)*cell] = (235, 230, 222)

    LEAF = (78, 184, 106)
    VEIN = (32, 96, 45)
    CORNER = (48, 64, 208)
    STEM = (48, 90, 138)
    PATTERN = [
        [3, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 2, 1, 1, 1, 0],
        [1, 1, 2, 2, 2, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1, 0],
        [0, 1, 1, 2, 1, 1, 0, 0],
        [0, 0, 1, 2, 1, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
    ]
    palette = {1: LEAF, 2: VEIN, 3: CORNER, 4: STEM}

    for j, row in enumerate(PATTERN):
        for i, val in enumerate(row):
            if val == 0:
                continue
            img[j*cell:(j+1)*cell, i*cell:(i+1)*cell] = palette[val]

    for k in range(9):
        c = min(k * cell, size - 1)
        cv2.line(img, (c, 0), (c, size - 1), (200, 198, 192), 1)
        cv2.line(img, (0, c), (size - 1, c), (200, 198, 192), 1)

    return img


def load_image(path: str | Path) -> np.ndarray:
    """Read a BGR image or raise if it cannot be decoded."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def build_output_path(
    image_path: str | Path,
    name: str,
    dst: str | Path | None = None,
) -> Path:
    """Return ``<stem>_<name><suffix>`` per the subject naming rule."""
    image_path = Path(image_path)
    out_dir = Path(dst) if dst is not None else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_path.stem}_{name}{image_path.suffix}"


def build_aug_output_path(
    image_path: str | Path,
    methods: tuple[str, ...],
    index: int,
    dst: str | Path | None = None,
) -> Path:
    """Return ``<stem>_<Method1>_<Method2>_<i><suffix>`` per decision #9.

    ``methods`` lists the augmentation methods in the order they were applied.
    """
    image_path = Path(image_path)
    out_dir = Path(dst) if dst is not None else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(methods)
    return out_dir / f"{image_path.stem}_{suffix}_{index}{image_path.suffix}"


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write ``image`` to ``path`` and return the path."""
    path = Path(path)
    cv2.imwrite(str(path), image)
    return path
