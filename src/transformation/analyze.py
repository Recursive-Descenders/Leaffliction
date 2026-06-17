from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from plantcv import plantcv as pcv  # type: ignore[import-not-found]

from transformation.leaf_cache import get_leaf_mask
from transformation.mask import build_mask
from transformation.util import (
    iter_image_paths,
    largest_leaf_mask,
    make_output_path,
    skip_image,
    validate_image_readable,
)


def apply_analyze(
    image: np.ndarray,
    leaf_mask: np.ndarray | None = None,
    *,
    image_path: Path | None = None,
) -> np.ndarray | None:
    if leaf_mask is None and image_path is not None:
        record = get_leaf_mask(image_path)
        if record is None:
            return None
        leaf_mask = record.mask
    if leaf_mask is None:
        leaf_mask = largest_leaf_mask(build_mask(image))
    if leaf_mask is None:
        return None

    labeled_mask, num_labels = pcv.create_labels(mask=leaf_mask)
    return pcv.analyze.size(
        img=image,
        labeled_mask=labeled_mask,
        n_labels=num_labels,
    )


def analyze(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="analyze"):
        try:
            image = validate_image_readable(image_path)
        except ValueError as exc:
            skip_image(image_path, str(exc))
            continue

        record = get_leaf_mask(image_path)
        if record is None:
            skip_image(image_path, "no detected leaf")
            continue

        analysis_image = apply_analyze(image, record.mask)
        if analysis_image is None:
            skip_image(image_path, "no detected leaf")
            continue

        output_file = make_output_path(
            src, dst, image_path, file, "analyze"
        )
        cv2.imwrite(str(output_file), analysis_image)
        saved_paths.append(output_file)

    return saved_paths
