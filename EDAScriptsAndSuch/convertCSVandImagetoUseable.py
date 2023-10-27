import os
import csv
import shutil

# Define the class mapping
class_mapping = {
    'A10': 'Fighters',
    'AV8B': 'Fighters',
    'EF2000': 'Fighters',
    'F4': 'Fighters',
    'F14': 'Fighters',
    'F15': 'Fighters',
    'F16': 'Fighters',
    'F18': 'Fighters',
    'F22': 'Fighters',
    'Rafale': 'Fighters',
    'F35': 'Fighters',
    'J20': 'Fighters',
    'JAS39': 'Fighters',
    'Mirage2000': 'Fighters',
    'Su34': 'Fighters',
    'Su57': 'Fighters',
    'Tornado': 'Fighters',
    'YF23': 'Fighters',
    'B1': 'Bombers',
    'B2': 'Bombers',
    'B52': 'Bombers',
    'F117': 'Bombers',
    'Mig31': 'Bombers',
    'SR71': 'Bombers',
    'Tu95': 'Bombers',
    'Tu160': 'Bombers',
    'Vulcan': 'Bombers',
    'XB70': 'Bombers',
    'A400M': 'Transport',
    'C2': 'Transport',
    'C5': 'Transport',
    'C17': 'Transport',
    'C130': 'Transport',
    'Be200': 'Transport',
    'US2': 'Transport',
    'V22': 'Transport',
    'E2': 'Surveillance',
    'E7': 'Surveillance',
    'P3': 'Surveillance',
    'RQ4': 'Surveillance',
    'U2': 'Surveillance',
    'AG600': 'Firefighting',
    'MQ9': 'Unmanned',
}

input_folder = "D:\Coding\School\img_rec_proj\dataset"

# Output folder for processed files
output_folder = 'D:\Coding\School\img_rec_proj\\EmptyData\\train'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):  # Assuming your images have a '.jpg' extension
        image_name = os.path.splitext(filename)[0]
        csv_file = os.path.join(input_folder, image_name + '.csv')

        if os.path.exists(csv_file):
            # Open and read the CSV file
            with open(csv_file, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                rows = list(csv_reader)

            if len(rows) == 2:
                row = rows[0]
                if 'class' in row:
                    original_class = row['class']
                    if original_class in class_mapping:
                        new_class = class_mapping[original_class]

                        # Create the output folder for the new class
                        class_folder = os.path.join(output_folder, new_class)
                        if not os.path.exists(class_folder):
                            os.makedirs(class_folder)

                        # Copy the image to the new folder with the updated class name
                        image_path = os.path.join(input_folder, filename)
                        new_image_path = os.path.join(class_folder, filename)
                        shutil.copy(image_path, new_image_path)
                        print(f"image written to {new_image_path}")
