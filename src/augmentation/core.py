"""
Augmentor — K=2 RandAugment-lite composition (decisions #1 + #10).

Each call to ``Augmentor.apply`` draws two distinct methods from the pool,
samples a random magnitude for every parameter within its calibrated range,
and applies the two methods in call-order. The method names used for the
most recent call are stored on ``last_methods`` so the IO layer can encode
them into the output filename per decision #9.
"""
from dataclasses import dataclass

import numpy as np

from augmentation.registry import POOL, Method


K = 2
"""Number of methods composed per output image (decision #1)."""


@dataclass
class Augmentor:
    """Stateful K=2 augmentor.

    The RNG is seeded for reproducibility (decision #11, default
    ``--seed 42``); pass ``seed=None`` for stochastic runs.
    """

    pool: list[Method]
    seed: int | None = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.last_methods: tuple[str, ...] = ()
        if len(self.pool) < K:
            raise ValueError(
                f"Augmentor pool needs >= K={K} methods, "
                f"got {len(self.pool)}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Draw K methods, sample magnitudes, apply in call-order."""
        indices = self._rng.choice(len(self.pool), size=K, replace=False)
        methods = [self.pool[i] for i in indices]

        result = image
        for method in methods:
            kwargs = {
                name: float(self._rng.uniform(lo, hi))
                for name, (lo, hi) in method.params.items()
            }
            result = method.func(result, **kwargs)

        self.last_methods = tuple(m.name for m in methods)
        return result


def default_augmentor(seed: int | None = 42) -> Augmentor:
    """Construct an Augmentor over the MVP geometric pool."""
    return Augmentor(pool=POOL, seed=seed)
