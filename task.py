import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Avoid division by zero
epsilon = 1e-10

# Function to compute vegetation indices
def compute_vegetation_indices(nir, red, green, swir):
    ndvi = (nir - red) / (nir + red + epsilon)
    savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5  # L = 0.5
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    gndvi = (nir - green) / (nir + green + epsilon)
    ndwi = (nir - swir) / (nir + swir + epsilon)

    return {
        "NDVI": ndvi,
        "SAVI": savi,
        "MSAVI": msavi,
        "GNDVI": gndvi,
        "NDWI": ndwi,
    }

# Function to load and process TIFF files
def process_tiff_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".tif") or file.endswith(".tiff"):
            file_path = os.path.join(input_folder, file)
            print(f"Processing: {file_path}")

            with rasterio.open(file_path) as src:
                nir = src.read(4).astype(float)   # NIR Band
                red = src.read(3).astype(float)   # Red Band
                green = src.read(2).astype(float) # Green Band
                swir = src.read(5).astype(float)  # SWIR Band

            # Compute vegetation indices
            veg_indices = compute_vegetation_indices(nir, red, green, swir)

            # Plot and save vegetation indices
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()

            for i, (name, index) in enumerate(veg_indices.items()):
                axes[i].imshow(index, cmap="jet")
                axes[i].set_title(name)
                axes[i].axis("off")
                plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=axes[i], fraction=0.046, pad=0.04)

                # Save each index as an image
                save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{name}.png")
                plt.imsave(save_path, index, cmap="jet")

            plt.tight_layout()
            plt.show()

    print(f"All vegetation index maps saved in: {output_folder}")

# Example Usage:
input_folder = "/content/drive/MyDrive/datasett/images"  # Update this path
output_folder = "/content/drive/MyDrive/datasett/output"  # Update this path
process_tiff_images(input_folder, output_folder)

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Avoid division by zero
epsilon = 1e-10

# Function to compute vegetation indices
def compute_vegetation_indices(nir, red, green, swir):
    ndvi = (nir - red) / (nir + red + epsilon)
    savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5  # L = 0.5
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    gndvi = (nir - green) / (nir + green + epsilon)
    ndwi = (nir - swir) / (nir + swir + epsilon)

    return {
        "NDVI": ndvi,
        "SAVI": savi,
        "MSAVI": msavi,
        "GNDVI": gndvi,
        "NDWI": ndwi,
    }

# Function to load and process TIFF files
def process_tiff_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".tif") or file.endswith(".tiff"):
            file_path = os.path.join(input_folder, file)
            print(f"Processing: {file_path}")

            with rasterio.open(file_path) as src:
                nir = src.read(4).astype(float)   # NIR Band
                red = src.read(3).astype(float)   # Red Band
                green = src.read(2).astype(float) # Green Band
                swir = src.read(5).astype(float)  # SWIR Band

            # Compute vegetation indices
            veg_indices = compute_vegetation_indices(nir, red, green, swir)

            # Plot and save the first five vegetation indices
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()

            for i, (name, index) in enumerate(veg_indices.items()):
                if i >= 5:  # Only plot first five indices
                    break
                axes[i].imshow(index, cmap="jet")
                axes[i].set_title(name)
                axes[i].axis("off")
                plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=axes[i], fraction=0.046, pad=0.04)

                # Save each index as an image
                save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{name}.png")
                plt.imsave(save_path, index, cmap="jet")

            plt.tight_layout()
            plt.show()

    print(f"All vegetation index maps saved in: {output_folder}")

# Example Usage:
input_folder = "/content/drive/MyDrive/datasett/images"  # Update this path
output_folder = "/content/drive/MyDrive/datasett/output"  # Update this path
process_tiff_images(input_folder, output_folder)

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to the uploaded TIFF mask file
file_path = "/content/drive/MyDrive/datasett/masks/VBWVA_2016_1_1_GeoTIFF_fractional_mask.tif"

# Read the TIFF file
with rasterio.open(file_path) as src:
    mask = src.read()  # Read all bands
    profile = src.profile

# If the mask has multiple bands, stack them as RGB
if mask.shape[0] >= 3:
    rgb_image = np.stack([mask[0], mask[1], mask[2]], axis=-1)
else:
    # If only one band, use a colormap
    rgb_image = mask[0]  # Use first band for grayscale display

# Display the colored mask image
plt.figure(figsize=(8, 6))
plt.imshow(rgb_image, cmap="jet")  # Apply a colormap
plt.colorbar()
plt.title("Labeled Mask Image")
plt.axis("off")
plt.show()

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to read and display TIFF mask images
def display_tiff_masks(folder_path):
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]

    if not tiff_files:
        print("No TIFF mask images found in the folder.")
        return

    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        print(f"Processing: {file_path}")

        with rasterio.open(file_path) as src:
            mask = src.read()  # Read all bands

        # If the mask has multiple bands, stack them as RGB
        if mask.shape[0] >= 3:
            rgb_image = np.stack([mask[0], mask[1], mask[2]], axis=-1)
        else:
            # If only one band, use a colormap
            rgb_image = mask[0]  # Use first band for grayscale display

        # Display the colored mask image
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image, cmap="jet")  # Apply a colormap
        plt.colorbar()
        plt.title(file)
        plt.axis("off")
        plt.show()

# Example usage
folder_path = "/content/drive/MyDrive/datasett/masks"  # Update with your actual folder path
display_tiff_masks(folder_path)

from google.colab import drive
drive.mount('/content/drive')



image_patch_size = 32



import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_indices(image_path):
    # Read the vegetation image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to float for calculations
    img = img.astype(np.float32) / 255.0

    # Split channels (Assuming RGB: Red = img[:,:,2], Green = img[:,:,1], Blue = img[:,:,0])
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    nir = green  # Assuming NIR is captured in the green channel (update if NIR is available separately)

    # Compute Vegetation Indices
    ndvi = (nir - red) / (nir + red + 1e-8)  # NDVI
    savi = ((nir - red) / (nir + red + 0.5)) * (1.5)  # SAVI (L=0.5)
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2  # MSAVI
    biomass = nir * (red / (nir + red + 1e-8))  # Biomass Index (approximation)

    # **Thresholding for Vegetation & Soil**
    vegetation_mask = (ndvi > 0.2).astype(np.uint8)  # Threshold for vegetation
    soil_mask = ((ndvi > -0.1) & (ndvi < 0.2)).astype(np.uint8)  # Threshold for soil

    # Convert masks to displayable format
    vegetation_display = vegetation_mask * 255
    soil_display = soil_mask * 255

    # Plot results
    indices = {'NDVI': ndvi, 'SAVI': savi, 'MSAVI': msavi, 'Biomass': biomass}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for ax, (name, index) in zip(axes[0], indices.items()):
        ax.imshow(index, cmap='RdYlGn')
        ax.set_title(name)
        ax.axis("off")

    # Display vegetation and soil masks
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")



    plt.show()

# Example Usage
image_path = "/content/drive/MyDrive/green-liquid-with-pink-foam_23-2147934190.jpg"  # Replace with your actual image path

import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_indices(image_path):
    # Read the vegetation image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return  # Exit the function if image loading failed

    # Convert to float for calculations
    img = img.astype(np.float32) / 255.0

    # ... (rest of your code) ...

import os
import cv2

dataset_root_folder = "/content/drive/MyDrive/datasett"  # Update with your actual dataset root
dataset_name = "/content/drive/MyDrive/datasett/images"  # Update with your actual dataset name

first_image = None  # Variable to store the first image

for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
    dir_name = path.split(os.path.sep)[-1]

    if dir_name == 'images':
        images = os.listdir(path)
        print(f"Found images in: {path}")
        print(images)

        for image_name in images:
            if image_name.endswith('.tiff'):
                print(f"Reading first image: {image_name}")
                image_path = os.path.join(path, image_name)
                first_image = cv2.imread(image_path, 1)  # Read the first .tiff image
                break  # Stop after reading the first image

        if first_image is not None:
            break  # Exit the outer loop after reading the first image

import os
import cv2

dataset_root_folder = "/content/drive/MyDrive/datasett"  # Update with your actual dataset root
dataset_name = "images"  # Folder containing images

first_image = None  # Variable to store the first image

# Walk through the dataset directory
for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
    dir_name = path.split(os.path.sep)[-1]

    if dir_name == 'images':
        images = os.listdir(path)
        print(f"Found images in: {path}")
        print(images)

        for image_name in images:
            if image_name.endswith('.tiff') or image_name.endswith('.tif'):
                print(f"Reading first image: {image_name}")
                image_path = os.path.join(path, image_name)
                first_image = cv2.imread(image_path, 1)  # Read the first .tiff image
                break  # Stop after reading the first image

        if first_image is not None:
            break  # Exit the outer loop after reading the first image

# Check if an image was loaded
if first_image is not None:
    print("First image successfully read!")
    cv2.imshow("First Image", first_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No TIFF image found.")

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Avoid division by zero
epsilon = 1e-10
image_patch_size = 66  # Define patch size

# Function to compute vegetation indices
def compute_vegetation_indices(nir, red, green, swir):
    ndvi = (nir - red) / (nir + red + epsilon)
    savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5  # L = 0.5
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    gndvi = (nir - green) / (nir + green + epsilon)
    ndwi = (nir - swir) / (nir + swir + epsilon)

    return {
        "NDVI": ndvi,
        "SAVI": savi,
        "MSAVI": msavi,
        "GNDVI": gndvi,
        "NDWI": ndwi,
    }

# Function to divide images into patches and process them
def process_tiff_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(".tif") or f.endswith(".tiff")]
    print(f"Total number of images in dataset: {len(image_files)}")  # Print total images

    ndvi_info = {}  # Dictionary to store NDVI shape and dimensions

    for file in image_files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing: {file_path}")

        with rasterio.open(file_path) as src:
            nir = src.read(4).astype(float)   # NIR Band
            red = src.read(3).astype(float)   # Red Band
            green = src.read(2).astype(float) # Green Band
            swir = src.read(5).astype(float)  # SWIR Band

        # Compute vegetation indices for the entire image
        veg_indices = compute_vegetation_indices(nir, red, green, swir)

        # Get image dimensions
        height, width = nir.shape
        num_patches_x = height // image_patch_size
        num_patches_y = width // image_patch_size

        print(f"Image Shape: {nir.shape}, Processing in {num_patches_x}×{num_patches_y} patches.")

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x_start, x_end = i * image_patch_size, (i + 1) * image_patch_size
                y_start, y_end = j * image_patch_size, (j + 1) * image_patch_size

                patch_indices = {name: index[x_start:x_end, y_start:y_end] for name, index in veg_indices.items()}

                # Store NDVI shape and dimensions for each patch
                patch_shape = patch_indices["NDVI"].shape
                patch_ndim = patch_indices["NDVI"].ndim
                patch_key = f"{file}_patch_{i}_{j}"
                ndvi_info[patch_key] = (patch_shape, patch_ndim)

                print(f"Patch ({i},{j}) Shape: {patch_shape}, Dimensions: {patch_ndim}D")

                # Save vegetation indices for the patch
                for name, patch in patch_indices.items():
                    save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{name}_patch_{i}_{j}.png")
                    plt.imsave(save_path, patch, cmap="jet")

    print("\nNDVI Shape and Dimensions for all patches:")
    for file, (shape, ndim) in ndvi_info.items():
        print(f"{file}: Shape {shape}, {ndim}D")

# Example Usage:
input_folder = "/content/drive/MyDrive/datasett/images"  # Update this path
output_folder = "/content/drive/MyDrive/datasett/output"  # Update this path
process_tiff_images(input_folder, output_folder)

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Avoid division by zero
epsilon = 1e-10
image_patch_size = 66  # Define patch size

# Function to compute vegetation indices
def compute_vegetation_indices(nir, red, green, swir):
    ndvi = (nir - red) / (nir + red + epsilon)
    savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5  # L = 0.5
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    gndvi = (nir - green) / (nir + green + epsilon)
    ndwi = (nir - swir) / (nir + swir + epsilon)

    return {
        "NDVI": ndvi,
        "SAVI": savi,
        "MSAVI": msavi,
        "GNDVI": gndvi,
        "NDWI": ndwi,
    }

# Function to divide images into patches and process them
def process_tiff_images(input_folder, output_folder, csv_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(".tif") or f.endswith(".tiff")]
    print(f"Total number of images in dataset: {len(image_files)}")  # Print total images

    ndvi_info = {}  # Dictionary to store NDVI shape and dimensions
    image_dataset = []  # List to store image patches
    patch_details = []  # List to store patch information

    minmax_scaler = MinMaxScaler()  # Normalization

    for file in image_files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing: {file_path}")

        with rasterio.open(file_path) as src:
            nir = src.read(4).astype(float)   # NIR Band
            red = src.read(3).astype(float)   # Red Band
            green = src.read(2).astype(float) # Green Band
            swir = src.read(5).astype(float)  # SWIR Band

        # Compute vegetation indices for the entire image
        veg_indices = compute_vegetation_indices(nir, red, green, swir)

        # Get image dimensions
        height, width = nir.shape
        num_patches_x = height // image_patch_size
        num_patches_y = width // image_patch_size

        print(f"Image Shape: {nir.shape}, Processing in {num_patches_x}×{num_patches_y} patches.")

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x_start, x_end = i * image_patch_size, (i + 1) * image_patch_size
                y_start, y_end = j * image_patch_size, (j + 1) * image_patch_size

                patch_indices = {name: index[x_start:x_end, y_start:y_end] for name, index in veg_indices.items()}

                # Store NDVI shape and dimensions for each patch
                patch_shape = patch_indices["NDVI"].shape
                patch_ndim = patch_indices["NDVI"].ndim
                patch_key = f"{file}_patch_{i}_{j}"
                ndvi_info[patch_key] = (patch_shape, patch_ndim)

                # Normalize patches
                normalized_patches = {name: minmax_scaler.fit_transform(patch) for name, patch in patch_indices.items()}

                # Append to dataset list
                image_dataset.append(normalized_patches)

                # Save patches as NumPy arrays
                for name, patch in normalized_patches.items():
                    npy_save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{name}_patch_{i}_{j}.npy")
                    np.save(npy_save_path, patch)

                # Save vegetation indices for the patch as images
                for name, patch in patch_indices.items():
                    img_save_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{name}_patch_{i}_{j}.png")
                    plt.imsave(img_save_path, patch, cmap="jet")

                # Store patch information in CSV
                patch_details.append({
                    "File": file,
                    "Patch": f"Patch_{i}_{j}",
                    "Shape": patch_shape,
                    "Dimensions": patch_ndim
                })

    # Save patch details to CSV
    df = pd.DataFrame(patch_details)
    df.to_csv(csv_file, index=False)

    print("\nNDVI Shape and Dimensions for all patches:")
    for file, (shape, ndim) in ndvi_info.items():
        print(f"{file}: Shape {shape}, {ndim}D")

# Example Usage:
input_folder = "/content/drive/MyDrive/datasett/images"  # Update this path
output_folder = "/content/drive/MyDrive/datasett/output"  # Update this path
csv_file = "/content/drive/MyDrive/datasett/patch_details.csv"  # CSV to save patch info

process_tiff_images(input_folder, output_folder, csv_file)

{
    "NDVI": np.array(...),  # First patch NDVI values
    "SAVI": np.array(...),  # First patch SAVI values
    "MSAVI": np.array(...),  # First patch MSAVI values
    "GNDVI": np.array(...),  # First patch GNDVI values
    "NDWI": np.array(...),  # First patch NDWI values
}



import random

def display_random_patches(output_folder, num_patches=5):
    patch_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    random_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    fig, axes = plt.subplots(1, len(random_patches), figsize=(15, 5))

    for ax, patch_file in zip(axes, random_patches):
        patch_path = os.path.join(output_folder, patch_file)
        patch_image = plt.imread(patch_path)
        ax.imshow(patch_image)
        ax.set_title(patch_file.split("_")[1])  # Show index name
        ax.axis("off")

    plt.suptitle("Random Patch Samples")
    plt.show()

# Call function to display random patches
display_random_patches(output_folder)

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

# Function to display random patches
def display_random_patches(output_folder, num_patches=5):
    patch_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    random_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    fig, axes = plt.subplots(1, len(random_patches), figsize=(15, 5))

    for ax, patch_file in zip(axes, random_patches):
        patch_path = os.path.join(output_folder, patch_file)
        patch_image = plt.imread(patch_path)
        ax.imshow(patch_image)
        ax.set_title(patch_file.split("_")[1])  # Show index name
        ax.axis("off")

    plt.suptitle("Random Patch Samples")
    plt.show()

import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt

# Define multiple color variations for each class
background_colors = [(204, 0, 0)]  # Background color
vegetation_colors = [
    (125, 255, 122), (96, 255, 151), (83, 255, 164), (115, 255, 131),
    (109, 255, 138), (112, 255, 135), (102, 255, 144), (99, 255, 148),
    (51, 255, 164), (60, 255, 157), (35, 255, 196), (28, 255, 202),
    (12, 244, 235), (19, 255, 222), (36, 255, 209), (31, 255, 206)
]
soil_colors = [(226, 169, 41)]  # Soil color

# Function to convert RGB values to label
def rgb_to_label(rgb):
    rgb_tuple = tuple(rgb)

    if rgb_tuple in background_colors:
        return 0  # Background
    elif rgb_tuple in vegetation_colors:
        return 1  # Vegetation
    elif rgb_tuple in soil_colors:
        return 2  # Soil
    else:
        return 255  # Unknown class

# Simulated mask dataset (Replace this with actual mask images)
# Assuming mask_dataset is a NumPy array of shape (num_images, height, width, 3)
mask_dataset = np.random.randint(0, 256, (10, 66, 66, 3), dtype=np.uint8)  # Example random images

labels = []
for i in range(mask_dataset.shape[0]):
    mask_image = mask_dataset[i]
    label_image = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)

    for x in range(mask_image.shape[0]):
        for y in range(mask_image.shape[1]):
            label_image[x, y] = rgb_to_label(mask_image[x, y])

    labels.append(label_image)

print("Total processed labels:", len(labels))

# Function to divide an image into 66x66 patches
def create_patches(image, patch_size=66):
    patches = []
    h, w = image.shape[-2:]  # Get height and width

    # Ensure dimensions are divisible by patch_size
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size

    # Iterate over the image to extract valid patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches

# Function to read and display TIFF mask patches
def display_tiff_patches(folder_path, patch_size=66):
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]

    if not tiff_files:
        print("No TIFF mask images found in the folder.")
        return

    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        print(f"\nProcessing: {file_path}")

        with rasterio.open(file_path) as src:
            mask = src.read()  # Read all bands

        print(f"Original Mask Shape: {mask.shape}")  # (Bands, Height, Width)

        # Generate patches (ensuring all are 66x66)
        patches = create_patches(mask, patch_size=patch_size)
        print(f"Total patches created: {len(patches)}")

        # Compute individual_patched_mask and assign it to label
        individual_patched_mask = patches
        label = individual_patched_mask

        # Display some patches
        for idx, patch in enumerate(patches[:15]):  # Show first 5 patches
            plt.figure(figsize=(3, 3))
            if patch.shape[0] >= 3:  # If at least 3 bands, show as RGB
                patch_display = np.stack([patch[0], patch[1], patch[2]], axis=-1)
            else:
                patch_display = patch[0]  # Single-band grayscale

            plt.imshow(patch_display, cmap="jet")
            plt.title(f"Patch {idx+1}")
            plt.axis("off")
            plt.show()

            print(f"Patch {idx+1} Values:\n", patch)

# Example usage
folder_path = "/content/drive/MyDrive/datasett/masks"  # Update with your actual folder path
display_tiff_patches(folder_path)

# Example Usage:
input_folder = "/content/drive/MyDrive/datasett/images"  # Update this path
output_folder = "/content/drive/MyDrive/datasett/output"  # Update this path
csv_file = "/content/drive/MyDrive/datasett/patch_details.csv"  # CSV to save patch info

process_tiff_images(input_folder, output_folder, csv_file)

import random

def display_random_patches(output_folder, num_patches=5):
    patch_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    random_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    fig, axes = plt.subplots(1, len(random_patches), figsize=(15, 5))

    for ax, patch_file in zip(axes, random_patches):
        patch_path = os.path.join(output_folder, patch_file)
        patch_image = plt.imread(patch_path)
        ax.imshow(patch_image)
        ax.set_title(patch_file.split("_")[1])  # Show index name
        ax.axis("off")

    plt.suptitle("Random Patch Samples")
    plt.show()

# Call function to display random patches
display_random_patches(output_folder)

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

# Function to display random patches
def display_random_patches(output_folder, num_patches=5):
    patch_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    random_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    fig, axes = plt.subplots(1, len(random_patches), figsize=(15, 5))

    for ax, patch_file in zip(axes, random_patches):
        patch_path = os.path.join(output_folder, patch_file)
        patch_image = plt.imread(patch_path)
        ax.imshow(patch_image)
        ax.set_title(patch_file.split("_")[1])  # Show index name
        ax.axis("off")

    plt.suptitle("Random Patch Samples")
    plt.show()

    return [os.path.join(output_folder, f) for f in random_patches]

# Function to segment colors using K-Means clustering
def segment_colors_kmeans(image_path, num_clusters=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))  # Resize for processing

    # Reshape image into (num_pixels, 3) format for clustering
    pixels = image_resized.reshape(-1, 3)

    # Apply K-Means clustering to segment colors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Create segmented image
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_resized.shape).astype(np.uint8)

    # Display original and segmented images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_resized)
    ax[0].set_title("Original Patch")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title("Segmented Patch")
    ax[1].axis("off")

    plt.show()

# Function to define and train a simple CNN for segmentation
def train_cnn_for_segmentation():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax'),  # 3 output classes for color segmentation
        Reshape((1, 1, 3))  # Reshape for visualization
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main execution
# Main execution
output_folder = "/content/drive/MyDrive/datasett/output"  # Corrected path to the output folder
random_patches = display_random_patches(output_folder)


# Apply color segmentation on each selected patch
for patch in random_patches:
    segment_colors_kmeans(patch)

# Train CNN (if labeled data is available)
cnn_model = train_cnn_for_segmentation()

import random
import numpy as np

def display_random_patches_with_values(output_folder, num_patches=5):
    patch_files = [f for f in os.listdir(output_folder) if f.endswith(".npy")]
    random_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    for patch_file in random_patches:
        patch_path = os.path.join(output_folder, patch_file)
        patch_data = np.load(patch_path)  # Load patch values

        print(f"\nPatch: {patch_file}")
        print(f"Mean: {np.mean(patch_data):.4f}, Min: {np.min(patch_data):.4f}, Max: {np.max(patch_data):.4f}")
        print(f"Patch Values:\n{patch_data}\n")

        # Display the patch
        plt.figure(figsize=(5, 5))
        plt.imshow(patch_data, cmap="jet")
        plt.title(patch_file)
        plt.colorbar()
        plt.show()

# Call function to display random patches with values
display_random_patches_with_values(output_folder)

print("Array Shape:",image_dataset.shape)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_sample_image():
    """Generate a synthetic vegetation image with soil and background."""
    img = np.zeros((200, 300, 3), dtype=np.float32)

    # Simulated soil (brownish region)
    img[50:150, 50:150] = [0.5, 0.3, 0.1]  # RGB (Brownish)

    # Simulated vegetation (Green area)
    img[50:150, 160:260] = [0.2, 0.7, 0.2]  # RGB (Greenish)

    # Simulated background (Black/No Data)
    img[:50, :] = 0  # Black top
    img[150:, :] = 0  # Black bottom

    return img

def compute_indices(img):
    """Compute NDVI and other vegetation indices from an RGB image."""
    # Extract channels
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    nir = green  # Simulated NIR from Green (Modify if real NIR exists)

    # Compute Vegetation Indices
    ndvi = (nir - red) / (nir + red + 1e-8)  # NDVI
    gndvi = (nir - green) / (nir + green + 1e-8)  # GNDVI
    savi = ((nir - red) / (nir + red + 0.5)) * (1.5)  # SAVI (L=0.5)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))  # EVI
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2  # MSAVI

    # Plot results
    indices = {'NDVI': ndvi, 'GNDVI': gndvi, 'SAVI': savi, 'EVI': evi, 'MSAVI': msavi}
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (name, index) in zip(axes, indices.items()):
        ax.imshow(index, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    plt.show()

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to read and display TIFF mask images
def display_tiff_masks(folder_path):
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]

    if not tiff_files:
        print("No TIFF mask images found in the folder.")
        return

    for file in tiff_files:
        file_path = os.path.join(folder_path, file)
        print(f"Processing: {file_path}")

        with rasterio.open(file_path) as src:
            mask = src.read()  # Read all bands

        # If the mask has multiple bands, stack them as RGB
        if mask.shape[0] >= 3:
            rgb_image = np.stack([mask[0], mask[1], mask[2]], axis=-1)
        else:
            # If only one band, use a colormap
            rgb_image = mask[0]  # Use first band for grayscale display

        # Display the colored mask image
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image, cmap="jet")  # Apply a colormap
        plt.colorbar()
        plt.title(file)
        plt.axis("off")
        plt.show()

# Example usage
folder_path = "/content/drive/MyDrive/datasett/masks"  # Update with your actual folder path
display_tiff_masks(folder_path)

from google.colab import drive
drive.mount('/content/drive')



image_patch_size = 32



import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_indices(image_path):
    # Read the vegetation image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to float for calculations
    img = img.astype(np.float32) / 255.0

    # Split channels (Assuming RGB: Red = img[:,:,2], Green = img[:,:,1], Blue = img[:,:,0])
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    nir = green  # Assuming NIR is captured in the green channel (update if NIR is available separately)

    # Compute Vegetation Indices
    ndvi = (nir - red) / (nir + red + 1e-8)  # NDVI
    savi = ((nir - red) / (nir + red + 0.5)) * (1.5)  # SAVI (L=0.5)
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2  # MSAVI
    biomass = nir * (red / (nir + red + 1e-8))  # Biomass Index (approximation)

    # **Thresholding for Vegetation & Soil**
    vegetation_mask = (ndvi > 0.2).astype(np.uint8)  # Threshold for vegetation
    soil_mask = ((ndvi > -0.1) & (ndvi < 0.2)).astype(np.uint8)  # Threshold for soil

    # Convert masks to displayable format
    vegetation_display = vegetation_mask * 255
    soil_display = soil_mask * 255

    # Plot results
    indices = {'NDVI': ndvi, 'SAVI': savi, 'MSAVI': msavi, 'Biomass': biomass}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for ax, (name, index) in zip(axes[0], indices.items()):
        ax.imshow(index, cmap='RdYlGn')
        ax.set_title(name)
        ax.axis("off")

    # Display vegetation and soil masks
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")



    plt.show()

from skimage.metrics import structural_similarity as ssim

def compute_similarity(original, segmented):
    """Compute similarity scores between original and segmented images."""
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_gray - segmented_gray) ** 2)

    # Compute Structural Similarity Index (SSIM)
    ssim_score = ssim(original_gray, segmented_gray, data_range=segmented_gray.max() - segmented_gray.min())

    return mse, ssim_score

def segment_colors_kmeans(image_path, num_clusters=3):
    """Perform K-Means segmentation and compute similarity scores."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))

    # Reshape image for clustering
    pixels = image_resized.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Reconstruct segmented image
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_resized.shape).astype(np.uint8)

    # Compute similarity metrics
    mse, ssim_score = compute_similarity(image_resized, segmented_image)

    # Display images and scores
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_resized)
    ax[0].set_title("Original Patch")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title(f"Segmented Patch\nMSE: {mse:.2f}, SSIM: {ssim_score:.2f}")
    ax[1].axis("off")

    plt.show()

    return mse, ssim_score

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

def compute_scores(original_img, segmented_img, vegetation_mask, soil_mask):
    # Convert images to grayscale for comparison
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    segmented_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

    # Flatten masks for IoU calculation
    vegetation_mask_flat = vegetation_mask.flatten()
    soil_mask_flat = soil_mask.flatten()
    segmented_flat = segmented_gray.flatten()

    # Compute IoU for vegetation and soil
    iou_vegetation = jaccard_score(vegetation_mask_flat, segmented_flat, average='binary')
    iou_soil = jaccard_score(soil_mask_flat, segmented_flat, average='binary')

    # Compute Dice Coefficient
    dice_vegetation = 2 * np.sum(vegetation_mask & segmented_gray) / (np.sum(vegetation_mask) + np.sum(segmented_gray) + 1e-8)
    dice_soil = 2 * np.sum(soil_mask & segmented_gray) / (np.sum(soil_mask) + np.sum(segmented_gray) + 1e-8)

    # Compute Pixel Accuracy
    pixel_accuracy = np.sum(segmented_gray == original_gray) / original_gray.size

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_gray - segmented_gray) ** 2)

    print(f"IoU Vegetation: {iou_vegetation:.4f}")
    print(f"IoU Soil: {iou_soil:.4f}")
    print(f"Dice Vegetation: {dice_vegetation:.4f}")
    print(f"Dice Soil: {dice_soil:.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"MSE: {mse:.4f}")

    # Display images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(segmented_img, cmap="gray")
    axes[0, 1].set_title("Segmented Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(vegetation_mask, cmap="gray")
    axes[1, 0].set_title("Vegetation Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(soil_mask, cmap="gray")
    axes[1, 1].set_title("Soil Mask")
    axes[1, 1].axis("off")

    plt.show()

# Example usage
original_image = cv2.imread("original_patch.png")  # Replace with actual image
segmented_image = cv2.imread("segmented_patch.png")  # Replace with segmented result
vegetation_mask = cv2.imread("vegetation_mask.png", 0)  # Load vegetation mask in grayscale
soil_mask = cv2.imread("soil_mask.png", 0)  # Load soil mask in grayscale

compute_scores(original_image, segmented_image, vegetation_mask, soil_mask)



from skimage.metrics import structural_similarity as ssim

def compute_similarity(original, segmented):
    """Compute similarity scores between original and segmented images."""
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_gray - segmented_gray) ** 2)

    # Compute Structural Similarity Index (SSIM)
    ssim_score = ssim(original_gray, segmented_gray, data_range=segmented_gray.max() - segmented_gray.min())

    return mse, ssim_score

def segment_colors_kmeans(image_path, num_clusters=3):
    """Perform K-Means segmentation and compute similarity scores."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))

    # Reshape image for clustering
    pixels = image_resized.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Reconstruct segmented image
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_resized.shape).astype(np.uint8)

    # Compute similarity metrics
    mse, ssim_score = compute_similarity(image_resized, segmented_image)

    # Display images and scores
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_resized)
    ax[0].set_title("Original Patch")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title(f"Segmented Patch\nMSE: {mse:.2f}, SSIM: {ssim_score:.2f}")
    ax[1].axis("off")

    plt.show()

    return mse, ssim_score

# Apply color segmentation and compute similarity
for patch in random_patches:
    mse, ssim_score = segment_colors_kmeans(patch)
    print(f"Patch: {patch} | MSE: {mse:.4f} | SSIM: {ssim_score:.4f}")

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

# Function to load images
def load_image(image_path, resize_shape=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, resize_shape)  # Resize to a fixed shape
    return img

# Function to compute evaluation metrics for a pair of images
def evaluate_segmentation(groundtruth_path, segmented_path):
    # Load images
    groundtruth = load_image(groundtruth_path)
    segmented = load_image(segmented_path)

    # Flatten images for computation
    groundtruth_flat = groundtruth.flatten()
    segmented_flat = segmented.flatten()

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(groundtruth_flat, segmented_flat, labels=[0, 255]).ravel()

    # Compute metrics
    accuracy = accuracy_score(groundtruth_flat, segmented_flat)
    mse = mean_squared_error(groundtruth_flat, segmented_flat)
    rmse = np.sqrt(mse)

    return tp, fp, fn, tn, accuracy, mse, rmse

# Function to evaluate an entire folder
def evaluate_folder(groundtruth_folder, segmented_folder):
    groundtruth_files = sorted([f for f in os.listdir(groundtruth_folder) if f.endswith(".png")])
    segmented_files = sorted([f for f in os.listdir(segmented_folder) if f.endswith(".png")])

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    accuracies, mses, rmses = [], [], []

    for gt_file, seg_file in zip(groundtruth_files, segmented_files):
        gt_path = os.path.join(groundtruth_folder, gt_file)
        seg_path = os.path.join(segmented_folder, seg_file)

        tp, fp, fn, tn, accuracy, mse, rmse = evaluate_segmentation(gt_path, seg_path)

        # Store results
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        accuracies.append(accuracy)
        mses.append(mse)
        rmses.append(rmse)

        print(f"Processed: {gt_file}")
        print(f"  Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Compute averages
    avg_accuracy = np.mean(accuracies)
    avg_mse = np.mean(mses)
    avg_rmse = np.mean(rmses)

    print("\n====== Overall Evaluation ======")
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}, Total TN: {total_tn}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")

# Example paths (Modify with actual paths)
groundtruth_folder = "/content/drive/MyDrive/datasett/masks"
segmented_folder = "/path/to/segmented_folder"

# Run evaluation for the entire folder
evaluate_folder(groundtruth_folder, segmented_folder)
