from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
) -> Path:
    if file is None:
        output_dir = dst / image_path.relative_to(src).parent
    else:
        output_dir = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
