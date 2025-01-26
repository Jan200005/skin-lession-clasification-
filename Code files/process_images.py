import os
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import exists
from PIL import Image
import extract_features as ef

# Assuming ef.extract_features and png_to_numpy_array are defined elsewhere
def png_to_numpy_array(file_path):
    # Open the image file
    image = Image.open(file_path).convert('L')  # convert to grayscale
    # Convert the image to a numpy array
    numpy_array = np.array(image)
    return numpy_array

# Paths and files
image_directory = "imgs_part_1"
path_mask = "masks"
file_features = '150features.csv'

# Get the list of all files in the image directory
all_files = os.listdir(image_directory)

# Filter the list to include only files with the desired extension (e.g., .png)
image_files = [f for f in all_files if f.endswith('.png')]

# Extract image IDs (filenames without extension)
image_id = [os.path.splitext(f)[0] for f in image_files]

# Assuming no need for labels here, so we'll skip that part

# Make list to store features
feature_names = ['Best Asymmetry', 'Mean Asymmetry', "Red1", "Red2", "White", "Black", "Light brown", "Dark brown", "Blue gray", "Has Veil?"]
features = []

# Create a list to store the image IDs that are actually processed
processed_image_ids = []

# Loop through the first 250 images
for i in range(250, 400):
    
    # Define filenames related to this image
    file_image = os.path.join(image_directory, image_files[i])
    maskname = image_id[i]
    file_mask = os.path.join(path_mask, f"{maskname}_mask.png")
    
    # Check if image file exists
    if not exists(file_image):
        print(f"Image file not found: {file_image}, skipping this image.")
        continue
    
    # Check if mask file exists
    if not exists(file_mask):
        print(f"Mask file not found for image {image_id[i]}, skipping this image.")
        continue
    
    # Read the image
    image = plt.imread(file_image)
    image_for_veil = cv2.imread(file_image, cv2.IMREAD_COLOR)
    mask = png_to_numpy_array(file_mask)
    
    # Measure features
    x = ef.extract_features(image, mask, image_for_veil)
       
    # Store in the list
    features.append(x)
    processed_image_ids.append(image_id[i])
    print(image_id[i])

# Convert the list of features to a NumPy array
features = np.array(features, dtype=np.float16)

# Filter out rows where all feature values are zero
non_zero_indices = np.where(np.any(features != 0, axis=1))[0]
features = features[non_zero_indices]
processed_image_ids = [processed_image_ids[i] for i in non_zero_indices]

# Create a DataFrame to save the image_id used + features
df_features = pd.DataFrame(features, columns=feature_names)

# Add the image_ids as the first column
df_features.insert(0, 'image_ids', processed_image_ids)

# Save to CSV file
df_features.to_csv(file_features, index=False)
