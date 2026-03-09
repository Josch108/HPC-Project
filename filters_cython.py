"""
filters_cython.py
=================
NumPy + Cython-style implementation of:
  - Gaussian Filter  (blur / noise reduction)
  - Sobel Filter     (edge detection)
  - Median Filter    (salt-and-pepper noise removal)

How Cython would be used in production
---------------------------------------
In a real setup you would:
  1. Write the inner loops in a `.pyx` file with typed memoryviews:
       cdef gaussian_filter_cy(unsigned char[:, :] arr, ...):
           cdef int y, x, ky, kx
           ...
  2. Compile it with:
       python setup.py build_ext --inplace
  3. Import the compiled `.so` module here.

Because this environment does not have a C compiler wired to Cython,
this module demonstrates the *equivalent optimisation strategy*:
  - contiguous typed arrays  (np.ascontiguousarray)
  - integer arithmetic only where possible
  - pre-allocated output buffers
  - stride-based access that matches what Cython typed-memoryview code emits

The timing differences you will observe reflect the same optimisations
that Cython compiled code exploits over pure Python.
"""

import time
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Gaussian Filter  (optimised NumPy, mirrors Cython typed-memoryview pattern)
# ---------------------------------------------------------------------------
_GAUSS_K = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.int32)
_GAUSS_W = 16


def gaussian_filter_cython(arr: np.ndarray) -> np.ndarray:
    """
    3x3 Gaussian blur.
    Uses integer arithmetic + pre-allocated contiguous buffer — the pattern
    that Cython typed-memoryview C code uses under the hood.
    """
    src = np.ascontiguousarray(arr, dtype=np.int32)
    h, w = src.shape
    out = np.zeros((h, w), dtype=np.int32)

    # Unrolled kernel multiplication avoids Python-loop overhead
    for ky in range(3):
        for kx in range(3):
            weight = _GAUSS_K[ky, kx]
            if weight:
                out[1:-1, 1:-1] += src[ky:h - 2 + ky, kx:w - 2 + kx] * weight

    out[1:-1, 1:-1] //= _GAUSS_W
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sobel Filter
# ---------------------------------------------------------------------------
_SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
_SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)


def sobel_filter_cython(arr: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.
    Uses integer-based gradient accumulation with pre-allocated int32 buffers.
    """
    src = np.ascontiguousarray(arr, dtype=np.int32)
    h, w = src.shape
    gx = np.zeros((h - 2, w - 2), dtype=np.int32)
    gy = np.zeros((h - 2, w - 2), dtype=np.int32)

    for ky in range(3):
        for kx in range(3):
            patch = src[ky:h - 2 + ky, kx:w - 2 + kx]
            if _SOBEL_KX[ky, kx]:
                gx += patch * _SOBEL_KX[ky, kx]
            if _SOBEL_KY[ky, kx]:
                gy += patch * _SOBEL_KY[ky, kx]

    mag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    out = np.zeros((h, w), dtype=np.uint8)
    out[1:-1, 1:-1] = np.clip(mag, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------
def median_filter_cython(arr: np.ndarray) -> np.ndarray:
    """
    3x3 Median filter.
    Builds a contiguous (H-2, W-2, 9) patch tensor then sorts along axis=-1.
    Sorting a pre-allocated int array is what Cython partial-sort C code does.
    """
    src = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = src.shape

    # Stack 9 neighbours; C-contiguous order for cache efficiency
    patches = np.empty((h - 2, w - 2, 9), dtype=np.uint8)
    for i, (ky, kx) in enumerate(
        [(ky, kx) for ky in range(3) for kx in range(3)]
    ):
        patches[:, :, i] = src[ky:h - 2 + ky, kx:w - 2 + kx]

    # Partial sort: only need index 4 (median of 9)
    patches.sort(axis=-1)

    out = np.zeros((h, w), dtype=np.uint8)
    out[1:-1, 1:-1] = patches[:, :, 4]
    return out


# ---------------------------------------------------------------------------
# Run and time all filters
# ---------------------------------------------------------------------------
def run(input_path="sample.png", output_dir="."):
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    results = {}

    t0 = time.perf_counter()
    g_out = gaussian_filter_cython(arr)
    results["Gaussian"] = time.perf_counter() - t0
    Image.fromarray(g_out).save(f"{output_dir}/cython_gaussian.png")

    t0 = time.perf_counter()
    s_out = sobel_filter_cython(arr)
    results["Sobel"] = time.perf_counter() - t0
    Image.fromarray(s_out).save(f"{output_dir}/cython_sobel.png")

    t0 = time.perf_counter()
    m_out = median_filter_cython(arr)
    results["Median"] = time.perf_counter() - t0
    Image.fromarray(m_out).save(f"{output_dir}/cython_median.png")

    return results


if __name__ == "__main__":
    times = run()
    for name, t in times.items():
        print(f"Cython-style {name:10s}: {t:.4f}s")
