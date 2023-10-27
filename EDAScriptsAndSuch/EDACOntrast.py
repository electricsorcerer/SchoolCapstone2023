import os
import cv2
import numpy as np

# Function to adjust contrast of an image
def adjust_contrast(image, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Define the input and output directories
input_directory = 'D:\Coding\School\img_rec_proj\EmptyData\composites'  
output_directory = 'D:\Coding\School\img_rec_proj\EmptyData\\contrast' 

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through files in the input directory
for filename in os.listdir(input_directory):
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    if filename.lower().endswith('_average.png'):
        # Load the average image
        average_image = cv2.imread(input_path)

        if average_image is not None:
            # Adjust the contrast
            high_contrast_image = adjust_contrast(average_image, alpha=1.5, beta=30)

            # Save the higher contrast image
            cv2.imwrite(output_path, high_contrast_image)
            print(f'Converted {filename} to higher contrast.')

print("All average images converted to higher contrast.")