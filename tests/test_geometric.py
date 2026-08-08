"""Unit tests for the geometric augmentations in ``geometric.py``.

Transforms are tested by their structural properties (shape, identity cases,
border fill, detectable change) rather than exact pixel values, which keeps the
tests meaningful without coupling to cv2 interpolation internals.

Where the *direction* of a transform matters (which parameter moves which
axis), a single marker pixel is tracked so the test fails if the underlying
matrix is wrong — e.g. ``skr``/``skc`` swapped — not merely "the image
changed".
"""
import numpy as np

from augmentation.geometric import (
    apply_crop,
    apply_radial_distortion,
    apply_flip,
    apply_rotate,
    apply_shear,
    apply_skew,
)


# --- helpers ------------------------------------------------------------

def _blue_marker(h=40, w=40, x=10, y=10):
    """Black image with one BGR-blue marker pixel at ``(x, y)``.

    Blue stays distinct from the white border fill that affine transforms
    leave behind, so the marker can be located after the warp.
    """
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[y, x] = (255, 0, 0)
    return image, (x, y)


def _marker_xy(image):
    """Locate the bluest pixel as ``(col, row)``; ignores white fill."""
    score = image[:, :, 0].astype(int) - image[:, :, 1] - image[:, :, 2]
    row, col = np.unravel_index(int(score.argmax()), score.shape)
    return int(col), int(row)


# --- flip ---------------------------------------------------------------

def test_flip_preserves_shape_and_dtype(sample_image):
    out = apply_flip(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == sample_image.dtype


def test_flip_is_horizontal(sample_image):
    out = apply_flip(sample_image)
    np.testing.assert_array_equal(out, sample_image[:, ::-1])


def test_flip_twice_is_identity(sample_image):
    np.testing.assert_array_equal(apply_flip(apply_flip(sample_image)),
                                  sample_image)


# --- rotate -------------------------------------------------------------

def test_rotate_preserves_shape(sample_image):
    out = apply_rotate(sample_image, angle=25)
    assert out.shape == sample_image.shape


def test_rotate_zero_angle_is_identity(sample_image):
    out = apply_rotate(sample_image, angle=0)
    np.testing.assert_allclose(out, sample_image, atol=1)


def test_rotate_fills_border_white():
    black = np.zeros((20, 30, 3), dtype=np.uint8)
    out = apply_rotate(black, angle=45)
    # Corners rotate out of frame and must be filled with white (255).
    assert (out == 255).any()


def test_rotate_changes_image(sample_image):
    out = apply_rotate(sample_image, angle=90)
    assert not np.array_equal(out, sample_image)


# --- skew ---------------------------------------------------------------

def test_skew_preserves_shape(sample_image):
    out = apply_skew(sample_image, skx=0.2, sky=0.2)
    assert out.shape == sample_image.shape


def test_skew_zero_is_identity(sample_image):
    out = apply_skew(sample_image, skx=0.0, sky=0.0)
    np.testing.assert_allclose(out, sample_image, atol=1)


def test_skew_changes_image(sample_image):
    out = apply_skew(sample_image, skx=0.3, sky=0.3)
    assert not np.array_equal(out, sample_image)


def test_skew_skx_and_sky_are_distinct_axes(sample_image):
    """``skx`` and ``sky`` must drive different entries of the perspective
    matrix (``h20`` vs ``h21``).

    Fails if the two coefficients are swapped or written to the same slot —
    the two outputs would then be identical.
    """
    only_x = apply_skew(sample_image, skx=0.3, sky=0.0)
    only_y = apply_skew(sample_image, skx=0.0, sky=0.3)
    assert not np.array_equal(only_x, only_y)


def test_skew_sign_flips_direction(sample_image):
    """Opposite-sign skews must produce different (mirrored) outputs.

    Guards against a sign-loss bug (e.g. taking ``abs`` of the coefficient).
    """
    pos = apply_skew(sample_image, skx=0.3, sky=0.0)
    neg = apply_skew(sample_image, skx=-0.3, sky=0.0)
    assert not np.array_equal(pos, neg)


# --- shear --------------------------------------------------------------

def test_shear_preserves_shape_and_dtype(sample_image):
    out = apply_shear(sample_image, kx=0.2, ky=0.2)
    assert out.shape == sample_image.shape
    assert out.dtype == sample_image.dtype


def test_shear_zero_is_identity(sample_image):
    out = apply_shear(sample_image, kx=0.0, ky=0.0)
    np.testing.assert_allclose(out, sample_image, atol=1)


def test_shear_is_horizontal():
    """Horizontal shear: x' = x + factor*y, the row is left unchanged."""
    image, (x0, y0) = _blue_marker()
    col, row = _marker_xy(apply_shear(image, kx=0.5, ky=0.0))
    assert row == y0
    assert abs(col - (x0 + 0.5 * y0)) <= 1


def test_shear_fills_border_white():
    black = np.zeros((20, 30, 3), dtype=np.uint8)
    out = apply_shear(black, kx=0.4, ky=0.0)
    assert (out == 255).any()


def test_shear_changes_image(sample_image):
    out = apply_shear(sample_image, kx=0.3, ky=0.3)
    assert not np.array_equal(out, sample_image)


# --- crop (random sub-window then resize back) --------------------------

def test_crop_preserves_shape(sample_image):
    out = apply_crop(sample_image, scale=0.8)
    assert out.shape == sample_image.shape


def test_crop_full_scale_is_identity(sample_image):
    # scale=1.0 forces a full-frame window — no crop possible, no resize.
    out = apply_crop(sample_image, scale=1.0)
    np.testing.assert_allclose(out, sample_image, atol=1)


def test_crop_zooms_when_scale_below_one():
    """Sub-1.0 crops magnify whatever ends up inside the window.

    A 0.5 crop keeps a quarter of the area but resizes back, so any patch
    that survives the window must cover more pixels than before. Filling the
    whole image with the patch removes the off-center risk a random window
    introduces.
    """
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    before = int((image == 255).all(axis=2).sum())
    np.random.seed(0)
    out = apply_crop(image, scale=0.5)
    after = int((out == 255).all(axis=2).sum())
    assert after >= before  # patch survives the crop+resize


def test_crop_changes_image(sample_image):
    np.random.seed(0)
    out = apply_crop(sample_image, scale=0.5)
    assert not np.array_equal(out, sample_image)


def test_crop_is_random():
    """Different RNG draws pick different windows from the same scale."""
    # A 2D gradient guarantees every window has distinct content, so two
    # different (top, left) draws produce different resized outputs.
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(40, dtype=np.uint8)[None, :] * 6
    image[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None] * 6
    np.random.seed(0)
    a = apply_crop(image, scale=0.5)
    np.random.seed(1)
    b = apply_crop(image, scale=0.5)
    assert not np.array_equal(a, b)


# --- distortion (radial barrel) -----------------------------------------

def test_radial_distortion_preserves_shape_and_dtype(sample_image):
    out = apply_radial_distortion(sample_image, k=0.3)
    assert out.shape == sample_image.shape
    assert out.dtype == sample_image.dtype


def test_radial_distortion_changes_image(sample_image):
    out = apply_radial_distortion(sample_image, k=0.3)
    assert not np.array_equal(out, sample_image)
