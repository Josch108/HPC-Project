"""
filters_numpy.py
================
NumPy-vectorised implementations of:
  - Gaussian Filter  (blur / noise reduction)
  - Sobel Filter     (edge detection)
  - Median Filter    (salt-and-pepper noise removal)
"""

import time
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------
GAUSSIAN_KERNEL = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=np.float32) / 16.0


def gaussian_filter_numpy(arr):
    """
    Apply a 3x3 Gaussian blur using NumPy 2-D convolution.
    Uses a manual sliding-window approach with array slices (no scipy).
    """
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)

    # Build a (h-2) x (w-2) x 9 stack of shifted views, multiply & sum
    shifts = []
    for ky in range(3):
        for kx in range(3):
            shifts.append(arr[ky:h - 2 + ky, kx:w - 2 + kx] *
                          GAUSSIAN_KERNEL[ky, kx])

    out[1:-1, 1:-1] = np.sum(shifts, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sobel Filter
# ---------------------------------------------------------------------------
SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _apply_kernel(arr, kernel):
    """Convolve a 3x3 kernel over arr using NumPy slice sums."""
    h, w = arr.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            out += arr[ky:h - 2 + ky, kx:w - 2 + kx] * kernel[ky, kx]
    return out


def sobel_filter_numpy(arr):
    """Apply Sobel edge detection using NumPy slice-based convolution."""
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.uint8)
    gx = _apply_kernel(arr.astype(np.float32), SOBEL_KX)
    gy = _apply_kernel(arr.astype(np.float32), SOBEL_KY)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    out[1:-1, 1:-1] = np.clip(mag, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------
def median_filter_numpy(arr):
    """
    Apply a 3x3 Median filter using NumPy.
    Constructs a (H-2) x (W-2) x 9 matrix of patch values and takes median.
    """
    h, w = arr.shape
    # Stack 9 shifted views along a new axis
    patches = np.stack([
        arr[ky:h - 2 + ky, kx:w - 2 + kx]
        for ky in range(3) for kx in range(3)
    ], axis=-1)  # shape: (H-2, W-2, 9)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[1:-1, 1:-1] = np.median(patches, axis=-1).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Run and time all filters
# ---------------------------------------------------------------------------
def run(input_path="sample.png", output_dir="."):
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    results = {}

    # Gaussian
    t0 = time.perf_counter()
    g_out = gaussian_filter_numpy(arr)
    results["Gaussian"] = time.perf_counter() - t0
    Image.fromarray(g_out).save(f"{output_dir}/numpy_gaussian.png")

    # Sobel
    t0 = time.perf_counter()
    s_out = sobel_filter_numpy(arr)
    results["Sobel"] = time.perf_counter() - t0
    Image.fromarray(s_out).save(f"{output_dir}/numpy_sobel.png")

    # Median
    t0 = time.perf_counter()
    m_out = median_filter_numpy(arr)
    results["Median"] = time.perf_counter() - t0
    Image.fromarray(m_out).save(f"{output_dir}/numpy_median.png")

    return results


if __name__ == "__main__":
    times = run()
    for name, t in times.items():
        print(f"NumPy {name:10s}: {t:.4f}s")
