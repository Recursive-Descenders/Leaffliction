"""
Method pool registry.

The Augmentor draws K=2 methods from this pool per output image
(decision #1 in docs/decision-graph.md). Each ``params`` map records sampling
range for every keyword argument, calibrated per EDA decision #2:

  - High freedom (B2 shape is NOT the classification signal):
    flip / rotate / shear / skew / crop / radial_distortion → defaults.
  - Low freedom (B1 color IS the signal):
    hue / saturation → tightest range; deferred until S3 (color_jitter) ships.
  - Forbidden (B3 features scattered):
    large-area erasure is rejected outright.

MVP pool (decision #8 — revisit-planned): geometric methods only.
"""
from dataclasses import dataclass, field
from typing import Callable

from augmentation.geometric import (
    apply_crop,
    apply_flip,
    apply_radial_distortion,
    apply_rotate,
    apply_shear,
    apply_skew,
)


@dataclass(frozen=True)
class Method:
    """A pool entry. ``params`` maps each kwarg to its safe sampling range."""

    name: str
    func: Callable
    params: dict[str, tuple[float, float]] = field(default_factory=dict)


POOL: list[Method] = [
    Method("Flip", apply_flip),
    Method("Rotate", apply_rotate, {"angle": (-25.0, 25.0)}),
    Method(
        "Shear",
        apply_shear,
        {"kx": (-0.2, 0.2), "ky": (-0.2, 0.2)},
    ),
    Method(
        "Skew",
        apply_skew,
        {"skx": (-0.3, 0.3), "sky": (-0.3, 0.3)},
    ),
    Method("Crop", apply_crop, {"scale": (0.7, 1.0)}),
    Method("Distortion", apply_radial_distortion, {"k": (-0.2, 0.2)}),
]
"""Geometric-only MVP pool. Append color_jitter once S3 ships (decision #8)."""
