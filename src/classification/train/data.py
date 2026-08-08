"""Load feature CSVs and split data per class for training."""

from pathlib import Path

import numpy as np
import pandas as pd

METADATA_COLUMNS = ("image_path", "label")
TRAIN_RATIO = 0.8


def load_features_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {path}")
    return df, feature_columns


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    *,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        shuffled = class_df.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**31)),
        )
        n_train = int(len(shuffled) * train_ratio)
        if n_train == 0 and len(shuffled) > 0:
            n_train = 1

        train_parts.append(shuffled.iloc[:n_train])
        test_parts.append(shuffled.iloc[n_train:])

    return pd.concat(train_parts), pd.concat(test_parts)


def dataframe_to_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    return X, y
