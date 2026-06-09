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

BLUE = (255, 0, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 79, 255)
DOT_RADIUS = 5


def _draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for point in landmarks:
        x = int(point[0, 0])
        y = int(point[0, 1])
        cv2.circle(image, (x, y), DOT_RADIUS, color, -1)


def apply_pseudolandmarks(image: np.ndarray) -> np.ndarray | None:
    leaf_mask = largest_leaf_mask(build_mask(image))
    if leaf_mask is None:
        return None

    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=image.copy(),
        mask=leaf_mask,
    )
    if (
        not isinstance(left, np.ndarray)
        or not isinstance(right, np.ndarray)
        or not isinstance(center_h, np.ndarray)
    ):
        return None

    landmark_image = image.copy()
    _draw_landmarks(landmark_image, left, BLUE)
    _draw_landmarks(landmark_image, right, MAGENTA)
    _draw_landmarks(landmark_image, center_h, ORANGE)
    return landmark_image


def pseudolandmarks(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(
        src, file, desc="pseudolandmarks"
    ):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        landmark_image = apply_pseudolandmarks(image)
        if landmark_image is None:
            print(f"Skipped image without pseudolandmarks: {image_path}")
            continue

        output_file = make_output_path(
            src, dst, image_path, file, "pseudolandmarks"
        )
        cv2.imwrite(str(output_file), landmark_image)
        saved_paths.append(output_file)

    return saved_paths
