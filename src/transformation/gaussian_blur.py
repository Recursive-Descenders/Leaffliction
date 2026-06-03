from pathlib import Path

import cv2  # type: ignore[import-not-found]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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

    if file is None:
        if not src.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {src}")
        image_paths = [
            path
            for path in src.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        image_path = Path(file)
        if not image_path.is_absolute():
            image_path = src / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        image_paths = [image_path]

    saved_paths = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        blurred_image = cv2.GaussianBlur(image, (strength, strength), 0)

        if file is None:
            relative_path = image_path.relative_to(src)
            output_path = dst / relative_path.parent
        else:
            output_path = dst

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_path / f"{image_path.stem}_blur{image_path.suffix}"
        )
        cv2.imwrite(str(output_file), blurred_image)
        saved_paths.append(output_file)

    return saved_paths
