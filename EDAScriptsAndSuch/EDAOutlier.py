import os
import cv2
import numpy as np
from scipy import stats
import shutil

image_directory = "D:\Coding\School\img_rec_proj\EmptyData\\train"

outlier_folder = 'D:\Coding\School\img_rec_proj\outlier_images'
os.makedirs(outlier_folder, exist_ok=True)

# Initialize variables to store image brightness values and filenames
brightness_values = []
file_names = []

# Loop through the images to calculate brightness values
for folder in os.listdir(image_directory):
    for filename in os.listdir("D:\Coding\School\img_rec_proj\EmptyData\\train\\"+folder):
        if filename.endswith(".jpg"):  
            img = cv2.imread(os.path.join(image_directory, folder, filename))
            if img is not None:
                # Convert the image to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Calculate the average brightness (intensity) of the image
                avg_brightness = np.mean(gray_img)
                brightness_values.append(avg_brightness)
                file_names.append(os.path.join(image_directory, folder, filename))
                print(f"{os.path.join(image_directory, folder, filename)}")

# Calculate Z-scores for brightness values
z_scores = np.abs(stats.zscore(brightness_values))

# Set a threshold for outlier detection (e.g., z-score > 2.0)
threshold = 3.5

# Identify outliers
outliers = np.where(z_scores > threshold)[0]

# Print and save the outlier images
if len(outliers) > 0:
    print("Outliers found in the following images:")
    for outlier_index in outliers:
        print(f"Image: {file_names[outlier_index]} Value: {brightness_values[outlier_index]}")
        # Copy the outlier image to the outlier folder
        shutil.copy(file_names[outlier_index], os.path.join(outlier_folder, os.path.basename(file_names[outlier_index])))
else:
    print("No outliers found in the dataset.")