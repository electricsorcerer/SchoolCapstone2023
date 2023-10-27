import os
import cv2
import numpy as np

# Function to calculate the average image
def calculate_average_image(image_paths, output_path):
    image_sum = np.zeros((300, 400, 3), dtype=np.float64)
    image_count = 0

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.resize(image, (400, 300))
            image_sum += image
            image_count += 1

    if image_count > 0:
        avg_image = (image_sum / image_count).astype(np.uint8)
        cv2.imwrite(output_path, avg_image)
        print(f'Created average image for {os.path.basename(os.path.dirname(image_paths[0]))}')

# Define the input and output directories
input_directory = "D:\Coding\School\img_rec_proj\EmptyData\\train"
output_directory = "D:\Coding\School\img_rec_proj\EmptyData\\composites"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through subfolders in the input directory
for folder_name in os.listdir(input_directory):
    folder_path = os.path.join(input_directory, folder_name)

    if os.path.isdir(folder_path):
        # List of images in the subfolder
        image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if image_paths:
            output_path = os.path.join(output_directory, f'{folder_name}_average.png')
            calculate_average_image(image_paths, output_path)

print("All average images created.")