"""Training pipeline: extract, split, forest, evaluate."""

from pathlib import Path

import numpy as np
import typer

from classification.train.data import (
    dataframe_to_xy,
    load_features_csv,
    stratified_split,
)
from classification.train.evaluate import evaluate
from classification.train.extract import extract, save_features_csv
from classification.train.forest import (
    MODEL_FILENAME,
    build_forest,
    save_forest,
)
from transformation.util import (
    validate_dst_is_directory,
    validate_source_exists,
)

SOURCE_DIR = Path("leaves/images")
DEFAULT_DST = Path("outputs/classification/train")
FEATURES_FILENAME = "features_rf.csv"


def _ensure_features_csv(src: Path, csv_path: Path) -> None:
    if csv_path.exists():
        return

    validate_source_exists(src)
    if not src.is_dir():
        raise NotADirectoryError(
            f"Source must be a directory of class folders: {src}"
        )

    df = extract(src=src)
    save_features_csv(df, csv_path)
    print(f"Extracted: {len(df)} leaves")
    print(f"Saved CSV: {csv_path}")


def run_training(
    *,
    src: Path = SOURCE_DIR,
    dst: Path = DEFAULT_DST,
) -> None:
    validate_dst_is_directory(dst)
    dst.mkdir(parents=True, exist_ok=True)

    csv_path = dst / FEATURES_FILENAME
    _ensure_features_csv(src, csv_path)

    df, feature_columns = load_features_csv(csv_path)
    rng = np.random.default_rng()
    train_df, test_df = stratified_split(df, rng=rng)

    X_train, y_train = dataframe_to_xy(train_df, feature_columns)
    X_test, y_test = dataframe_to_xy(test_df, feature_columns)

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    forest = build_forest(
        X_train,
        y_train,
        feature_columns,
        rng=rng,
    )
    model_path = save_forest(forest, dst / MODEL_FILENAME)
    print(f"Saved model: {model_path}")

    evaluate(forest, X_test, y_test, dst=dst)


def run(
    src: Path = typer.Option(
        SOURCE_DIR,
        "-s",
        "--src",
        help=(
            "Image directory with class subfolders "
            f"(default: {SOURCE_DIR}). "
            "Used only when features CSV is missing. "
            "Example: 'uv run tr -s leaves/images -d out/train'"
        ),
    ),
    dst: Path = typer.Option(
        DEFAULT_DST,
        "-d",
        "--dst",
        help=(
            "Destination directory for training outputs "
            f"(default: {DEFAULT_DST})."
        ),
    ),
) -> None:
    try:
        run_training(src=src, dst=dst)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
