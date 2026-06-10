"""
Geometric augmentations.
"""
import numpy as np
import cv2


def apply_flip(image: np.ndarray) -> np.ndarray:
    """Horizontal mirror — reverse the pixel columns."""
    return cv2.flip(image, 1)  # flipCode=1 for horizontal flip


def apply_rotate(image: np.ndarray, angle: float = 25.0) -> np.ndarray:
    """Rotate the image by a specified angle."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated


def apply_skew(
    image: np.ndarray,
    skx: float = 0.001,
    sky: float = 0.008,
) -> np.ndarray:
    """Skew the image by specified factors along rows and columns."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    h20 = skx / w
    h21 = sky / h

    T_to_center = np.array(
        [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32
    )
    P = np.array(
        [[1, 0, 0], [0, 1, 0], [h20, h21, 1]], dtype=np.float32
    )
    T_back = np.array(
        [[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32
    )

    M = T_back @ P @ T_to_center
    skewed = cv2.warpPerspective(
        image, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return skewed


def apply_shear(image: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Shear the image by specified factors along rows and columns.

    Positive ``kx`` shears rightward, negative leftward; ``ky`` shears
    downward / upward.
    """
    h, w = image.shape[:2]
    M = np.array([[1, kx, 0], [ky, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(
        image, M, (w, h),  # type: ignore
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return sheared


def apply_crop(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Crop a sub-region then resize back to the original size."""
    h, w = image.shape[:2]
    ch, cw = int(h * scale), int(w * scale)
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    cropped = image[top:top + ch, left:left + cw]
    resized = cv2.resize(cropped, (w, h))
    return resized


def apply_radial_distortion(
    image: np.ndarray, k: float = 0.3
) -> np.ndarray:
    """Lens-like radial warp. ``k > 0`` shrinks toward center (corners go
    white); ``k < 0`` bulges outward. Magnitude grows with ``r²`` so the
    center stays put while edges warp the most.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_index, x_index = np.indices((h, w), dtype=np.float32)

    dx = x_index - cx
    dy = y_index - cy
    r2_norm = (dx**2 + dy**2) / (cx**2 + cy**2)
    distortion = k * r2_norm
    map_x = (x_index + dx * distortion).astype(np.float32)
    map_y = (y_index + dy * distortion).astype(np.float32)

    distorted = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return distorted
