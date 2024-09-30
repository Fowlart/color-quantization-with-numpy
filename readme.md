# Image Color Quantization with NumPy

This Python code demonstrates the fundamentals of **color quantization** using NumPy.  It provides a hands-on learning experience to explore:

* **Quantization:** The process of reducing the number of colors in an image by mapping pixels to a limited set of representative colors (codebook).
* **Euclidean Norm:** The calculation of distances between pixel colors and codebook colors using the Euclidean norm.
* **Broadcasting in NumPy:** How broadcasting facilitates efficient operations between arrays of different shapes.

## How it Works

1. **Image Loading:** 
   * The code loads an image (e.g., 'img556.jpg') and normalizes its pixel values to the range [0, 1].

2. **Quantization:**
   * A fixed codebook of colors is defined.
   * The image is reshaped into a 2D array where each row represents a pixel's RGB values.
   * Broadcasting is used to efficiently calculate the Euclidean distances between each pixel and all codebook colors.
   * Each pixel is then assigned the color from the codebook that is closest to it (based on the minimum Euclidean distance).
   * The quantized image is reconstructed using the codebook colors.

3. **Visualization and Analysis:**
   * The original and quantized images are displayed side-by-side.
   * The compression ratio is calculated to demonstrate the reduction in image size achieved through quantization.
   * Intermediate steps and calculations are printed for educational purposes, showcasing how broadcasting and Euclidean norms are utilized.

## Key Concepts

* **Vector Quantization (VQ):**  A technique for data compression that maps data points to a finite set of representative vectors (codebook).
* **Euclidean Norm:**  A measure of the distance between two points in Euclidean space, calculated as the square root of the sum of squared differences between corresponding elements.
* **Broadcasting:** A powerful NumPy feature that allows for efficient element-wise operations between arrays of different but compatible shapes.