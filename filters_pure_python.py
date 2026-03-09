"""
filters_pure_python.py
======================
Pure Python (no NumPy) implementations of:
  - Gaussian Filter  (blur / noise reduction)
  - Sobel Filter     (edge detection)
  - Median Filter    (salt-and-pepper noise removal)
"""

import math
import time
from PIL import Image


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------
GAUSSIAN_KERNEL = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]
GAUSSIAN_WEIGHT = 16  # sum of all kernel values


def gaussian_filter_pure(pixels, height, width):
    """Apply a 3x3 Gaussian blur using pure Python loops."""
    output = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            acc = 0
            for ky in range(3):
                for kx in range(3):
                    acc += pixels[y + ky - 1][x + kx - 1] * GAUSSIAN_KERNEL[ky][kx]
            output[y][x] = acc // GAUSSIAN_WEIGHT

    return output


# ---------------------------------------------------------------------------
# Sobel Filter
# ---------------------------------------------------------------------------
SOBEL_X = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
]

SOBEL_Y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
]


def sobel_filter_pure(pixels, height, width):
    """Apply Sobel edge detection using pure Python loops."""
    output = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = 0
            gy = 0
            for ky in range(3):
                for kx in range(3):
                    px = pixels[y + ky - 1][x + kx - 1]
                    gx += px * SOBEL_X[ky][kx]
                    gy += px * SOBEL_Y[ky][kx]
            magnitude = int(math.sqrt(gx * gx + gy * gy))
            output[y][x] = min(255, magnitude)

    return output


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------
def median_filter_pure(pixels, height, width):
    """Apply a 3x3 Median filter using pure Python loops."""
    output = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighborhood = []
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    neighborhood.append(pixels[y + ky][x + kx])
            neighborhood.sort()
            output[y][x] = neighborhood[4]  # median of 9 values

    return output


# ---------------------------------------------------------------------------
# Helper: PIL image -> 2-D list, and back
# ---------------------------------------------------------------------------
def image_to_2d(img):
    """Convert a grayscale PIL image to a 2-D list of ints."""
    width, height = img.size
    pixels = list(img.getdata())
    return [[pixels[y * width + x] for x in range(width)]
            for y in range(height)], height, width


def list2d_to_image(data, height, width):
    """Convert a 2-D list back to a grayscale PIL Image."""
    flat = [data[y][x] for y in range(height) for x in range(width)]
    img = Image.new("L", (width, height))
    img.putdata(flat)
    return img


# ---------------------------------------------------------------------------
# Run and time all filters
# ---------------------------------------------------------------------------
def run(input_path="sample.png", output_dir="."):
    img = Image.open(input_path).convert("L")
    pixels, height, width = image_to_2d(img)

    results = {}

    # Gaussian
    t0 = time.perf_counter()
    g_out = gaussian_filter_pure(pixels, height, width)
    results["Gaussian"] = time.perf_counter() - t0
    list2d_to_image(g_out, height, width).save(f"{output_dir}/pure_gaussian.png")

    # Sobel
    t0 = time.perf_counter()
    s_out = sobel_filter_pure(pixels, height, width)
    results["Sobel"] = time.perf_counter() - t0
    list2d_to_image(s_out, height, width).save(f"{output_dir}/pure_sobel.png")

    # Median
    t0 = time.perf_counter()
    m_out = median_filter_pure(pixels, height, width)
    results["Median"] = time.perf_counter() - t0
    list2d_to_image(m_out, height, width).save(f"{output_dir}/pure_median.png")

    return results


if __name__ == "__main__":
    times = run()
    for name, t in times.items():
        print(f"Pure Python {name:10s}: {t:.4f}s")
