from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _get_image_paths(
    src: Path,
    file: str | Path | None,
) -> list[Path]:
    if file is None:
        if not src.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {src}")
        return [
            path
            for path in src.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    image_path = Path(file)
    if not image_path.is_absolute():
        image_path = src / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    return [image_path]


def _build_mask(image: np.ndarray) -> np.ndarray:
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

    image_paths = _get_image_paths(src, file)
    saved_paths: list[Path] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        mask_image = _build_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=mask_image)

        if file is None:
            relative_path = image_path.relative_to(src)
            output_path = dst / relative_path.parent
        else:
            output_path = dst

        output_path.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_path / f"{image_path.stem}_mask{image_path.suffix}"
        )
        cv2.imwrite(str(output_file), masked_image)
        saved_paths.append(output_file)

    return saved_paths
