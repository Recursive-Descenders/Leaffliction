"""Random forest ensemble built from bootstrap-sampled decision trees."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from classification.train.tree import RandomTree, build_random_tree

N_TREES = 500
MODEL_FILENAME = "forest.joblib"


@dataclass(frozen=True)
class RandomForest:
    trees: list[RandomTree]
    classes_: np.ndarray
    feature_columns: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        class_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)

        for random_tree in self.trees:
            X_subset = X[:, random_tree.feature_indices]
            predictions = random_tree.tree.predict(X_subset)
            for row_index, label in enumerate(predictions):
                votes[row_index, class_to_index[label]] += 1

        return self.classes_[votes.argmax(axis=1)]


def build_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    *,
    n_trees: int = N_TREES,
    rng: np.random.Generator | None = None,
) -> RandomForest:
    if rng is None:
        rng = np.random.default_rng()

    trees: list[RandomTree] = []
    for _ in tqdm(
        range(n_trees),
        desc="forest",
        unit="tree",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
    ):
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        trees.append(build_random_tree(X[boot_idx], y[boot_idx], rng=rng))

    return RandomForest(
        trees=trees,
        classes_=np.unique(y),
        feature_columns=feature_columns,
    )


def save_forest(forest: RandomForest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, path)
    return path


def load_forest(path: Path) -> RandomForest:
    return joblib.load(path)
