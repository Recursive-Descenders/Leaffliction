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


def apply_roi(
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

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    leaf_roi = pcv.roi.rectangle(
        img=image,
        x=x,
        y=y,
        h=height,
        w=width,
    )

    selected_mask = pcv.roi.filter(
        mask=leaf_mask,
        roi=leaf_roi,
        roi_type="partial",
    )
    selected_contours, _ = cv2.findContours(
        selected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    roi_image = image.copy()
    cv2.drawContours(
        roi_image,
        selected_contours,
        -1,
        (0, 255, 0),
        cv2.FILLED,
    )
    cv2.drawContours(
        roi_image,
        leaf_roi.contours[0],
        -1,
        (255, 0, 0),
        3,
    )
    return roi_image


def roi(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="roi"):
        try:
            image = validate_image_readable(image_path)
        except ValueError as exc:
            skip_image(image_path, str(exc))
            continue

        record = get_leaf_mask(image_path)
        if record is None:
            skip_image(image_path, "no detected leaf")
            continue

        roi_image = apply_roi(image, record.mask)
        if roi_image is None:
            skip_image(image_path, "no detected leaf")
            continue

        output_file = make_output_path(
            src, dst, image_path, file, "roi"
        )
        cv2.imwrite(str(output_file), roi_image)
        saved_paths.append(output_file)

    return saved_paths
