"""Shared pytest fixtures for the augmentation tests.

Matplotlib is forced onto the headless ``Agg`` backend before any test imports
``visualization``, so the viewer can be exercised without opening a window.
"""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def sample_image() -> np.ndarray:
    """A small, deliberately asymmetric BGR image.

    The left and right halves differ so horizontal operations (flip, skew)
    produce a detectable change, and a single bright marker pixel lets a test
    follow where a transform sends a known point.
    """
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = (50, 100, 150)   # left half
    image[:, 15:] = (200, 60, 10)    # right half
    image[0, 0] = (255, 255, 255)    # top-left marker
    return image
