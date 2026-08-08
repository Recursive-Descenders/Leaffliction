"""Smoke tests for the augmentor pipeline (core / registry / preprocess)."""
from pathlib import Path

import numpy as np
import pytest

from augmentation.core import Augmentor, default_augmentor
from augmentation.preprocess import dedup_images, list_images
from augmentation.registry import POOL
from augmentation.util import build_aug_output_path


def test_pool_has_six_geometric_methods():
    names = [m.name for m in POOL]
    assert names == [
        "Flip", "Rotate", "Shear", "Skew", "Crop", "Distortion"
    ]


def test_augmentor_apply_returns_same_shape(sample_image):
    augmentor = default_augmentor(seed=42)
    result = augmentor.apply(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == sample_image.dtype


def test_augmentor_records_two_methods_per_apply(sample_image):
    augmentor = default_augmentor(seed=42)
    augmentor.apply(sample_image)
    assert len(augmentor.last_methods) == 2
    assert augmentor.last_methods[0] != augmentor.last_methods[1]


def test_augmentor_is_reproducible_with_same_seed(sample_image):
    a = default_augmentor(seed=42)
    b = default_augmentor(seed=42)
    out_a = a.apply(sample_image)
    out_b = b.apply(sample_image)
    assert np.array_equal(out_a, out_b)
    assert a.last_methods == b.last_methods


def test_augmentor_diverges_with_different_seeds(sample_image):
    a = default_augmentor(seed=1)
    b = default_augmentor(seed=2)
    seq_a = [a.apply(sample_image) for _ in range(5)]
    seq_b = [b.apply(sample_image) for _ in range(5)]
    assert any(not np.array_equal(x, y) for x, y in zip(seq_a, seq_b))


def test_augmentor_rejects_undersized_pool():
    with pytest.raises(ValueError, match="K=2"):
        Augmentor(pool=[POOL[0]], seed=42)


def test_build_aug_output_path_encodes_methods(tmp_path):
    src = tmp_path / "image (1).JPG"
    src.write_bytes(b"x")
    out = build_aug_output_path(
        src, ("Rotate", "Shear"), 3, dst=tmp_path / "out"
    )
    assert out.name == "image (1)_Rotate_Shear_3.JPG"
    assert out.parent == tmp_path / "out"
    assert out.parent.exists()


def _write_bytes(path: Path, blob: bytes) -> None:
    path.write_bytes(blob)


def test_dedup_drops_byte_identical_duplicates(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    _write_bytes(a, b"same")
    _write_bytes(b, b"different")
    _write_bytes(c, b"same")  # byte-identical to a

    kept, dropped = dedup_images([a, b, c])
    assert kept == [a, b]
    assert dropped == [c]


def test_list_images_filters_non_image_files(tmp_path):
    (tmp_path / "leaf.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "scan.PNG").write_bytes(b"x")
    found = [p.name for p in list_images(tmp_path)]
    assert sorted(found) == ["leaf.jpg", "scan.PNG"]
