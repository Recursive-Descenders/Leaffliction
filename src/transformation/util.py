from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_source_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {path}")


def validate_dst_is_directory(dst: Path) -> None:
    if dst.is_file():
        raise NotADirectoryError(
            f"Destination must be a directory, not a file: {dst}"
        )


def validate_image_readable(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension: {image_path.suffix} "
            f"(supported: {', '.join(sorted(IMAGE_EXTENSIONS))})"
        )
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def skip_image(image_path: Path, reason: str) -> None:
    print(f"Skipped {image_path.name}: {reason}")


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


def make_output_path(
    src: Path,
    dst: Path,
    image_path: Path,
    file: str | Path | None,
    suffix: str,
    *,
    extension: str | None = None,
) -> Path:
    if file is None:
        output_dir = dst / image_path.relative_to(src).parent
    else:
        output_dir = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    ext = extension if extension is not None else image_path.suffix
    return output_dir / f"{image_path.stem}_{suffix}{ext}"


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
