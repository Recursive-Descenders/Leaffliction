from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from plantcv import plantcv as pcv  # type: ignore[import-not-found]

from transformation.mask import build_mask
from transformation.util import (
    iter_image_paths,
    largest_leaf_mask,
    make_output_path,
)


def apply_analyze(image: np.ndarray) -> np.ndarray | None:
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
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        analysis_image = apply_analyze(image)
        if analysis_image is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        output_file = make_output_path(
            src, dst, image_path, file, "analyze"
        )
        cv2.imwrite(str(output_file), analysis_image)
        saved_paths.append(output_file)

    return saved_paths
