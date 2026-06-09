from collections.abc import Iterator
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_paths(
    src: Path,
    file: str | Path | None,
) -> list[Path]:
    if file is None:
        if not src.is_dir():
            raise NotADirectoryError(
                f"Source path is not a directory: {src}"
            )
        return [
            path
            for path in src.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    image_path = Path(file)
    if not image_path.is_absolute():
        image_path = src / image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file does not exist: {image_path}"
        )

    return [image_path]


def iter_image_paths(
    src: Path,
    file: str | Path | None,
    desc: str | None = None,
) -> Iterator[Path]:
    image_paths = get_image_paths(src, file)
    yield from tqdm(
        image_paths,
        desc=desc,
        unit="image",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
    )


def make_output_path(
    src: Path,
    dst: Path,
    image_path: Path,
    file: str | Path | None,
    suffix: str,
) -> Path:
    if file is None:
        output_dir = dst / image_path.relative_to(src).parent
    else:
        output_dir = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"


def largest_leaf_mask(mask_image: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros_like(mask_image)
    cv2.drawContours(leaf_mask, [largest_contour], -1, 255, -1)
    return leaf_mask
