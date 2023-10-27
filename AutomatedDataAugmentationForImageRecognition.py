import os
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw, ImageChops
import numpy as np

# Define a list of available image processing operations and their probabilities
available_operations = [
    ("FlipHorizontal", 0.3), 
    ("FlipVertical", 0.3),  
    ("Rotate90", 0.2),
    ("Malfunction", 0.5), 
    ("Rain", 0.3), 
    ("Contrast", 0.4),
    ("Crop", 0.4)
]
def select_operations():
    num_operations = random.randint(1, 3)  # Select up to four operations
    selected_operations = random.sample(
        [op for op, prob in available_operations],
        num_operations
    )
    return selected_operations

def convert_to_rgba(image):
    if image.mode == 'RGBA':
        return image  # Image is already in RGBA mode
    elif image.mode == 'RGB':
        return image.convert('RGBA')
    else:
        # Handle other image modes if needed
        return image


def process_image(image_path, input_root, output_root):
    try:
        edits = ""
        selected_operations = []
        # Determine the relative path from the input directory
        relative_path = os.path.relpath(image_path, input_root)

        # Construct the corresponding output directory path
        output_dir = os.path.join(output_root, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        try:
            image = Image.open(image_path)
            image = convert_to_rgba(image)
        except OSError:
            print(f"Error opening {image_path}. Skipping this image.")

        # Select four operations based on their probabilities
        selected_operations = select_operations()

        # Apply the selected operations to the image
        for operation in selected_operations:
                
            if operation == "Crop":
                img_width, img_height = image.size
                max_area_to_remove = 30 / 100.0 * img_height * img_width

                # Calculate the minimum dimensions based on the constraint
                min_width = int((max_area_to_remove / img_height) + 1)
                min_height = int((max_area_to_remove / img_width) + 1)

                # Generate random crop dimensions within the calculated constraints
                crop_width = random.randint(min_width, img_width)
                crop_height = random.randint(min_height, img_height)

                # Calculate random crop positions
                left = random.randint(0, img_width - crop_width)
                top = random.randint(0, img_height - crop_height)

                # Perform the crop
                image = image.crop((left, top, left + crop_width, top + crop_height))
                edits += "Crop, "

            elif operation == "Rotate90":
                image = image.transpose(Image.ROTATE_90)
                edits += "Rotate 90 degrees, "
            elif operation == "FlipHorizontal":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                edits += "Flip Horizontal, "
            elif operation == "FlipVertical":
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                edits += "Flip Vertical, "



            elif operation == "Malfunction":
                width, height = image.size
                alpha_range=(0.5, 0.9)

                # Create a NumPy array from the image
                img_array = np.array(image)

                # Generate random noise array with the same shape as the image
                noise = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)

                # Blend the original image with the noise
                alpha = random.uniform(alpha_range[0], alpha_range[1])
                noisy_image = (alpha * img_array + (1 - alpha) * noise).astype(np.uint8)

                # Create a PIL image from the noisy array
                noisy_image = Image.fromarray(noisy_image)
                edits += "Malfunction, "
            
            elif operation == "Rain":
                width, height = image.size
    
                # Define the maximum size of the rectangles within 1/5 of the image dimensions
                max_rect_width = width // 5
                max_rect_height = height // 5

                # Ensure that the mask size matches the image size
                mask = Image.new("L", (width, height), 255)  # Initialize with all ones

                for _ in range(random.randint(1, 4)):
                    # Generate random coordinates within the valid range
                    x1 = random.randint(0, width - max_rect_width)
                    y1 = random.randint(0, height - max_rect_height)
                    
                    # Ensure that x2 and y2 are within bounds and greater than x1 and y1
                    max_rect_width = min(max_rect_width, width - x1)
                    max_rect_height = min(max_rect_height, height - y1)
                    
                    if max_rect_width < 20 or max_rect_height < 20:
                        continue  # Skip this iteration if the available space is too small for a rectangle

                    rect_width = random.randint(20, max_rect_width)
                    rect_height = random.randint(20, max_rect_height)
                    
                    x2 = x1 + rect_width
                    y2 = y1 + rect_height

                    color = random.randint(0, 255)
                    draw = ImageDraw.Draw(mask)
                    draw.rectangle([(x1, y1), (x2, y2)], fill=0)  # Subtract the rectangle region from the mask

                # Apply a Gaussian blur to the image outside the regions defined by the random mask
                image_blurred = image.filter(ImageFilter.GaussianBlur(random.randint(6, 45)))

                # Create a masked image by combining the original image and the blurred image
                image = Image.composite(image, image_blurred, mask)

            elif operation == "Contrast":
                # Simulate fog by reducing image contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.9, 0.2))
                edits += "Contrast, "

        edits += " and nought else"
        print(relative_path + "| Edits: " + edits)

        # Convert the image to RGB mode before saving it as JPEG
        image = image.convert("RGB")

        # Save the processed image to the output directory with the same name
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        image.save(output_path, "JPEG")
        image.close()

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)} | {selected_operations}")


def process_images_in_directory(input_dir, output_dir):
    # Traverse subfolders and process images
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            process_image(image_path, input_dir, output_dir)

if __name__ == "__main__":
    input_directory = "D:\Coding\School\img_rec_proj\EmptyData\\train"  
    output_directory = "D:\Coding\School\img_rec_proj\\Output\\train" 

    process_images_in_directory(input_directory, output_directory)