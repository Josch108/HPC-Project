# HPC Project – Image Processing Filters Optimization

## Overview

This project was developed for the **High Performance Computing** course at Universidad Politécnica de Yucatán.  
The goal of the project is to evaluate the performance of different implementations of common image processing filters using Python, NumPy, and low-level optimization strategies.

The project compares three programming approaches:

- Pure Python (baseline)
- NumPy vectorized implementation
- Cython-style optimized implementation

The filters implemented are:

- Gaussian Filter (blur / smoothing)
- Sobel Filter (edge detection)
- Median Filter (salt-and-pepper noise removal)

The main objective is to measure execution time and analyze the speedup obtained by optimized methods.

---

## Project Structure

HPC-Project/
│
├── main.py
├── filters_pure_python.py
├── filters_numpy.py
├── filters_cython.py
├── output/
├── performance_analysis(1).ipynb
└── README.md


### Files

| File | Description |
|------|------------|
| main.py | Runs all filters, measures time, and saves results |
| filters_pure_python.py | Baseline implementation using loops |
| filters_numpy.py | Vectorized implementation using NumPy |
| filters_cython.py | Low-level optimized version (Cython-style) |
| output/ | Folder where generated images are saved |

---

## Requirements

Install dependencies:

pip install -r requirements.txt


---

## How to Run

Run with synthetic test image:

python main.py


Run with your own image:

python main.py path/to/image.png


The script will:

- Generate or load an image
- Apply all filters
- Measure execution time
- Save results in the output folder
- Print a performance comparison table

---

## Synthetic Test Image

If no image is provided, the program generates a 256×256 grayscale image containing:

- Gradient background
- Rectangle
- Circle
- Salt-and-pepper noise

This allows fair performance comparison.

---

## Implementations

### 1. Pure Python

- Uses nested loops
- Manual convolution
- Slowest implementation
- Used as baseline

### 2. NumPy

- Uses vectorized operations
- Uses optimized C backend
- Much faster than pure Python

### 3. Cython-style

- Uses contiguous memory arrays
- Uses integer arithmetic
- Preallocated buffers
- Simulates compiled C performance

---

## Performance Results

Typical results:

| Filter | Pure Python | NumPy | Cython-style | Speedup |
|--------|------------|--------|-------------|---------|
| Gaussian | Slow | Fast | Faster | ~19x |
| Sobel | Slow | Very fast | Very fast | ~70x |
| Median | Slow | Fast | Faster | ~15x |

Optimized implementations significantly reduce execution time.

---

## Output

The program saves images in:
output/


Examples:

- original.png
- gaussian_numpy.png
- sobel_numpy.png
- median_numpy.png
- gaussian_cython.png
- etc.

---

## Authors

Josué Octavio Chan Caballero  
Joel Alejandro Pérez Yupit  
Gael Alberto Lara Peña  

Data Engineering – Universidad Politécnica de Yucatán  
High Performance Computing

---

## License

This project is for academic purposes.