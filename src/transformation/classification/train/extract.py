from pathlib import Path
import random

import cv2  # type: ignore[import-not-found]
import pandas as pd
from tqdm import tqdm

from transformation.mask import LeafMaskResult, evaluate_leaf_mask
from transformation.util import IMAGE_EXTENSIONS

SOURCE_DIR = Path("leaves/images")
IMAGES_PER_DIR = 200
RANDOM_SEED = 42


def _label_from_path(image_path: Path) -> str:
    if len(image_path.parts) >= 2:
        return image_path.parent.name
    return image_path.stem


def _sample_image_paths(
    src: Path,
    *,
    per_dir: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue

        selected.extend(rng.sample(images, min(per_dir, len(images))))

    return selected


def _row_from_result(
    image_path: Path,
    result: LeafMaskResult,
) -> dict[str, object]:
    return {
        "image_path": str(image_path),
        "label": _label_from_path(image_path),
        "leaf_solidity": result.leaf_solidity,
        "mask_area_ratio": result.mask_area_ratio,
        "border_touch_ratio": result.border_touch_ratio,
    }


def extract_features(image_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for image_path in tqdm(
        image_paths,
        desc="extract",
        unit="image",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        result = evaluate_leaf_mask(image)
        if result is None:
            print(f"Skipped image without detected leaf: {image_path}")
            continue

        rows.append(_row_from_result(image_path, result))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract() -> pd.DataFrame:
    image_paths = _sample_image_paths(
        SOURCE_DIR,
        per_dir=IMAGES_PER_DIR,
        seed=RANDOM_SEED,
    )
    return extract_features(image_paths)


def main() -> None:
    df = extract()
    print(f"Extracted: {len(df)} leaves")


if __name__ == "__main__":
    main()
