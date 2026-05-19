from pathlib import Path
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from plantcv import plantcv as pcv  # type: ignore[import-not-found]
from transformation.mask import build_mask
from transformation.util import get_image_paths, make_output_path

MAGENTA = (255, 0, 255)
LINE_THICKNESS = 2


def _largest_leaf_mask(mask_image: np.ndarray) -> np.ndarray | None:
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


def analyze(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    image_paths = get_image_paths(src, file)
    saved_paths: list[Path] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        leaf_mask = _largest_leaf_mask(build_mask(image))
        if leaf_mask is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        labeled_mask, num_labels = pcv.create_labels(mask=leaf_mask)
        analysis_image = pcv.analyze.size(
            img=image,
            labeled_mask=labeled_mask,
            n_labels=num_labels,
        )

        output_file = make_output_path(
            src, dst, image_path, file, "analyze"
        )
        cv2.imwrite(str(output_file), analysis_image)
        saved_paths.append(output_file)

    return saved_paths
