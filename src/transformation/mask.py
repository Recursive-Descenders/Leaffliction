from pathlib import Path
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from transformation.util import iter_image_paths, make_output_path


def build_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Remove almost-black background.
    not_black = v_channel > 25

    # Leaf pixels usually have either color saturation or enough brightness.
    # Keeps green, yellow, brown, and damaged areas better than green-only HSV.
    colored_leaf = s_channel > 25

    # Restrict hue to natural leaf/disease range:
    # red/brown/yellow/green area in OpenCV HSV.
    hue_leaf_range = ((h_channel >= 0) & (h_channel <= 95))

    mask_bool = not_black & colored_leaf & hue_leaf_range
    mask_image = mask_bool.astype(np.uint8) * 255

    # Gentle cleanup only.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, open_kernel)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, close_kernel)

    return mask_image


def mask(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    saved_paths: list[Path] = []

    for image_path in iter_image_paths(src, file, desc="mask"):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        mask_image = build_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=mask_image)

        output_file = make_output_path(
            src, dst, image_path, file, "mask"
        )
        cv2.imwrite(str(output_file), masked_image)
        saved_paths.append(output_file)

    return saved_paths
