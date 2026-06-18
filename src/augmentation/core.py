"""
Augmentor — K=2 RandAugment-lite composition (decisions #1 + #10).

Each call to ``Augmentor.apply`` draws two distinct methods from the pool,
samples a random magnitude for every parameter within its calibrated range,
and applies the two methods in call-order. The method names used for the
most recent call are stored on ``last_methods`` so the IO layer can encode
them into the output filename per decision #9.

When ``fixed_methods`` is provided the random draw is skipped: the given
methods are applied in the specified order with magnitudes still sampled
randomly. This allows K=1 (single-method) runs.
"""
from dataclasses import dataclass, field

import numpy as np

from augmentation.registry import POOL, Method


K = 2
"""Default number of methods composed per output image (decision #1)."""


@dataclass
class Augmentor:
    """Stateful augmentor.

    The RNG is seeded for reproducibility (decision #11, default
    ``--seed 42``); pass ``seed=None`` for stochastic runs.

    When ``fixed_methods`` is set the random K-draw is bypassed and those
    methods are applied in order, so K can be any positive integer.
    """

    pool: list[Method]
    seed: int | None = 42
    fixed_methods: list[Method] | None = field(default=None)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.last_methods: tuple[str, ...] = ()
        if self.fixed_methods is None and len(self.pool) < K:
            raise ValueError(
                f"Augmentor pool needs >= K={K} methods, "
                f"got {len(self.pool)}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply methods to image, sampling magnitudes randomly."""
        if self.fixed_methods is not None:
            methods = self.fixed_methods
        else:
            indices = self._rng.choice(len(self.pool), size=K, replace=False)
            methods = [self.pool[i] for i in indices]

        # Reseed global numpy RNG from our seeded RNG so any method using
        # np.random.* (e.g., apply_crop's random offset) stays reproducible.
        np.random.seed(int(self._rng.integers(0, 2**31 - 1)))

        result = image
        for method in methods:
            kwargs = {
                name: float(self._rng.uniform(lo, hi))
                for name, (lo, hi) in method.params.items()
            }
            result = method.func(result, **kwargs)

        self.last_methods = tuple(m.name for m in methods)
        return result


def default_augmentor(
    seed: int | None = 42,
    fixed_methods: list[Method] | None = None,
) -> Augmentor:
    """Construct an Augmentor over the MVP geometric pool."""
    return Augmentor(pool=POOL, seed=seed, fixed_methods=fixed_methods)
