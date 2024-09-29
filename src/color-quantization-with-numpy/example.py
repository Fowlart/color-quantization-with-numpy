import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def _compute_euclidean_norm(vector: np.ndarray) -> np.float64:
    if vector.ndim != 1:
        raise ValueError("Input vector must be one-dimensional.")

    return np.sqrt(np.sum(np.square(vector)))


def quantize(image_array):

    height, width, _ = image_array.shape

    pixels = image_array.reshape(-1, 3)

    print(f"""
pixels.shape: {pixels.shape}
    
pixels.dtype: {pixels.dtype}
    
pixels.min: {pixels.min()}
    
pixels.mean: {pixels.mean()}
    
pixels.max: {pixels.max()}

pixel example: {pixels[165234, :]}
""")

    codebook = np.array([
        [0.0, 0.0, 0.0],  # Black
        [1.0, 1.0, 1.0],  # White
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue,
        [0.89803922, 0.79215686, 0.23921569] # color from picture
    ])

    print(f"""codebook: 
{codebook}

""")

    print(f"codebook.shape: {codebook.shape}","\n")

    # need to change shape, because broadcast will not work otherwise
    arr_dist = pixels[:, np.newaxis] - codebook

    # np.linalg.norm is used to calculate the distances between each image pixel
    # and the vectors from the codebook. This allows us to find the
    # closest codebook vector for each pixel,
    # which forms the basis for color quantization.
    euclidean_norms = np.linalg.norm(arr_dist, axis=2)

    # find an index of min Euclidean norm, for each pixel-color pair
    labels = np.argmin(euclidean_norms, axis=1)

    distance_for_one_pixel = arr_dist[[165234], :, :].astype(np.float32)

    # print for debug
    print(f"""
arr_dist.shape: {arr_dist.shape}
        
arr_dist example(165234 pixel):

distance_for_one_pixel.shape: {distance_for_one_pixel.shape}

{distance_for_one_pixel}""")

    print(f"""
arr_dist.min: {arr_dist.min()}, arr_dist.max: {arr_dist.max()}, arr_dist.mean: {arr_dist.mean()};
        
euclidean_norms.shape: {euclidean_norms.shape}, euclidean_norms.min: {euclidean_norms.min()}, 
       
euclidean_norms.max: {euclidean_norms.max()}, euclidean_norms.mean: {euclidean_norms.mean()}
        
euclidean_norms example: 
            
{euclidean_norms[[165234], :]}
            
labels: {labels}

label for pixel 165234: {labels[165234]}
""")

    _, y, _ = distance_for_one_pixel.shape
    print("Provide manual calculation for Euclidean norm for distance above:", "\n")
    for n in range(y):
        vector = distance_for_one_pixel[:, [n], :]
        print(vector)
        print(f"""euclidean_norm[{n}]: 
      {_compute_euclidean_norm(vector=(vector.flatten()))}

""")

    # Result: now image consist from codebook colors.
    # Color of each pixel of original image maximum close to color of
    # corresponding pixel from transformed image
    quantized_image = codebook[labels].reshape(height, width, 3)

    return quantized_image, codebook, height, width

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=4)

    path_to_img: str = os.path.join(os.curdir,"..","resources","Collage-of-Python-programming-aspects-syntax-libraries-Python-symbols-logo.jpg")

    # Load image
    image = Image.open(path_to_img)

    image_array = np.array(image) / 255.0

    # Quantize
    n_colors = 5
    quantized_image, codebook, height, width = quantize(image_array)

    # Display
    plt.imshow(quantized_image)
    plt.show()

    # Compression ratio
    original_size = image_array.nbytes
    compressed_size = (height * width) + codebook.nbytes
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}")