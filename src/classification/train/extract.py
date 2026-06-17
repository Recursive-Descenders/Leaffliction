from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from transformation.image_features import image_features_to_row
from transformation.lesion_cache import (
    ensure_lesion_record,
    lesion_record_to_image_features,
)
from transformation.util import IMAGE_EXTENSIONS, validate_source_exists

SOURCE_DIR = Path("leaves/images")
OUTPUT_CSV = Path("outputs/classification/train/features_rf.csv")


def _label_from_path(image_path: Path) -> str:
    if len(image_path.parts) >= 2:
        return image_path.parent.name
    return image_path.stem


def _all_image_paths(src: Path) -> list[Path]:
    selected: list[Path] = []

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        selected.extend(images)

    return selected


def _row_from_cache(image_path: Path) -> dict[str, object] | None:
    lesion_record = ensure_lesion_record(image_path)
    if lesion_record is None:
        return None

    features = lesion_record_to_image_features(lesion_record)
    return image_features_to_row(
        image_path,
        _label_from_path(image_path),
        features,
    )


def extract_features(image_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for image_path in tqdm(
        image_paths,
        desc="extract",
        unit="image",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
    ):
        row = _row_from_cache(image_path)
        if row is not None:
            rows.append(row)
            continue

        print(f"Skipped {image_path.name}: no detected leaf or spot mask")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract(
    *,
    src: Path = SOURCE_DIR,
) -> pd.DataFrame:
    image_paths = _all_image_paths(src)
    return extract_features(image_paths)


def save_features_csv(
    df: pd.DataFrame,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def run(
    src: Path = typer.Option(
        SOURCE_DIR,
        "-s",
        "--src",
        help=(
            "Image directory with class subfolders "
            f"(default: {SOURCE_DIR}). "
            "Example: 'uv run extract -s leaves/images -o features.csv'"
        ),
    ),
    output: Path = typer.Option(
        OUTPUT_CSV,
        "-o",
        "--output",
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    ),
) -> None:
    try:
        validate_source_exists(src)
        if not src.is_dir():
            raise NotADirectoryError(
                f"Source must be a directory of class folders: {src}"
            )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    df = extract(src=src)
    csv_path = save_features_csv(df, output)
    print(f"Extracted: {len(df)} leaves")
    print(f"Saved CSV: {csv_path}")
    print(
        "Masks reused from outputs/cache/leaf_mask/ "
        "and outputs/cache/lesion/"
    )


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
