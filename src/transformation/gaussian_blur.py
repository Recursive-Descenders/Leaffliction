from pathlib import Path
import cv2  # type: ignore[import-not-found]
from transformation.util import get_image_paths, make_output_path


def gaussian_blur(
    src: str | Path,
    dst: str | Path,
    file: str | Path | None = None,
    strength: int = 9,
) -> list[Path]:
    src = Path(src)
    dst = Path(dst)

    if strength <= 0 or strength % 2 == 0:
        raise ValueError("Blur strength must be a positive odd number")

    image_paths = get_image_paths(src, file)
    saved_paths: list[Path] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        blurred_image = cv2.GaussianBlur(image, (strength, strength), 0)

        output_file = make_output_path(
            src, dst, image_path, file, "blur"
        )
        cv2.imwrite(str(output_file), blurred_image)
        saved_paths.append(output_file)

    return saved_paths
