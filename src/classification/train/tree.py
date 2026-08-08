"""Random decision tree builder for random forest training."""

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

N_RANDOM_FEATURES = 5
MIN_SAMPLES_LEAF = 2


@dataclass(frozen=True)
class RandomTree:
    tree: DecisionTreeClassifier
    feature_indices: np.ndarray


def build_random_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> RandomTree:
    """Fit one tree on caller-supplied bootstrap rows and 5 random features.

    The caller prepares a bootstrap sample before calling, for example::

        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        tree = build_random_tree(X[boot_idx], y[boot_idx], rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )
    if X.shape[1] < N_RANDOM_FEATURES:
        raise ValueError(
            f"X must have at least {N_RANDOM_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    feature_indices = rng.choice(
        X.shape[1],
        size=N_RANDOM_FEATURES,
        replace=False,
    )
    X_subset = X[:, feature_indices]

    tree = DecisionTreeClassifier(
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=int(rng.integers(0, 2**31)),
    )
    tree.fit(X_subset, y)

    return RandomTree(tree=tree, feature_indices=feature_indices)
