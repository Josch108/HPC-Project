"""
main.py
=======
Orchestrates all three image-filter implementations, measures execution times,
saves filtered images, and prints a comparison table.

Usage:
    python main.py                   # uses built-in test image
    python main.py path/to/image.png
"""

import sys
import os
import time
import numpy as np
from PIL import Image, ImageDraw
import filters_pure_python as pure
import filters_numpy as npy
import filters_cython as cyt

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Create a synthetic test image if no real one is supplied
# ---------------------------------------------------------------------------
def make_test_image(path="sample.png", size=(256, 256)):
    """Generate a 256x256 grayscale test image with shapes and noise."""
    rng = np.random.default_rng(42)
    arr = np.zeros(size, dtype=np.uint8)

    # Background gradient
    for i in range(size[0]):
        arr[i, :] = int(i / size[0] * 128)

    # White rectangle
    arr[50:120, 50:120] = 220

    # Dark circle approximation
    cy, cx, r = 160, 160, 50
    for y in range(size[0]):
        for x in range(size[1]):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                arr[y, x] = 40

    # Salt-and-pepper noise
    noise_mask = rng.integers(0, 100, size=size)
    arr[noise_mask < 3] = 255   # salt
    arr[noise_mask > 97] = 0    # pepper

    img = Image.fromarray(arr, mode="L")
    img.save(path)
    print(f"  Test image created: {path}  ({size[0]}x{size[1]} px)")
    return path


# ---------------------------------------------------------------------------
# Timing wrapper
# ---------------------------------------------------------------------------
def time_filter(fn, *args, runs=1):
    elapsed = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed.append(time.perf_counter() - t0)
    return result, min(elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Input image ----
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "sample.png"
        if not os.path.exists(input_path):
            make_test_image(input_path)

    img = Image.open(input_path).convert("L")
    img.save(f"{OUTPUT_DIR}/original.png")
    print(f"\nImage loaded: {input_path}  size={img.size}")

    arr = np.array(img, dtype=np.uint8)
    pixels_2d, height, width = pure.image_to_2d(img)

    print("\nRunning filters …\n")

    timing = {
        "Gaussian": {},
        "Sobel": {},
        "Median": {},
    }

    # ---- Pure Python ----
    print("  [1/3] Pure Python …")
    r, t = time_filter(pure.gaussian_filter_pure, pixels_2d, height, width)
    pure.list2d_to_image(r, height, width).save(f"{OUTPUT_DIR}/pure_gaussian.png")
    timing["Gaussian"]["Pure Python"] = t

    r, t = time_filter(pure.sobel_filter_pure, pixels_2d, height, width)
    pure.list2d_to_image(r, height, width).save(f"{OUTPUT_DIR}/pure_sobel.png")
    timing["Sobel"]["Pure Python"] = t

    r, t = time_filter(pure.median_filter_pure, pixels_2d, height, width)
    pure.list2d_to_image(r, height, width).save(f"{OUTPUT_DIR}/pure_median.png")
    timing["Median"]["Pure Python"] = t

    # ---- NumPy ----
    print("  [2/3] NumPy …")
    r, t = time_filter(npy.gaussian_filter_numpy, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/numpy_gaussian.png")
    timing["Gaussian"]["NumPy"] = t

    r, t = time_filter(npy.sobel_filter_numpy, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/numpy_sobel.png")
    timing["Sobel"]["NumPy"] = t

    r, t = time_filter(npy.median_filter_numpy, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/numpy_median.png")
    timing["Median"]["NumPy"] = t

    # ---- Cython-style ----
    print("  [3/3] NumPy + Cython-style …")
    r, t = time_filter(cyt.gaussian_filter_cython, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/cython_gaussian.png")
    timing["Gaussian"]["Cython"] = t

    r, t = time_filter(cyt.sobel_filter_cython, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/cython_sobel.png")
    timing["Sobel"]["Cython"] = t

    r, t = time_filter(cyt.median_filter_cython, arr)
    Image.fromarray(r).save(f"{OUTPUT_DIR}/cython_median.png")
    timing["Median"]["Cython"] = t

    # ---- Print table ----
    header = f"\n{'Filter':<12} {'Pure Python':>14} {'NumPy':>12} {'Cython':>12}  {'Speedup (Py→NumPy)':>20}"
    print(header)
    print("-" * len(header))
    for filt, times in timing.items():
        py_t  = times["Pure Python"]
        np_t  = times["NumPy"]
        cy_t  = times["Cython"]
        speedup = py_t / np_t if np_t > 0 else float("inf")
        print(f"{filt:<12} {py_t:>13.4f}s {np_t:>11.4f}s {cy_t:>11.4f}s  {speedup:>18.1f}x")

    print(f"\nAll output images saved to: ./{OUTPUT_DIR}/")
    return timing


if __name__ == "__main__":
    main()
