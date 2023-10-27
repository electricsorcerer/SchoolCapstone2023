import tensorflow as tf
import cv2
import numpy as np

# Load the SavedModel
loaded_model = tf.saved_model.load('D:\Coding\School\img_rec_proj\TheModel')

# Load and preprocess the image
image_path = 'D:\Coding\School\img_rec_proj\Output\\train\C5\\02be7b39398493666593021f7d797c6f_0.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Resize to match the model's input shape
image = image / 255.0  # Normalize the image

# Convert the data type to float32
input_data = tf.convert_to_tensor(image, dtype=tf.float32)
input_data = tf.expand_dims(input_data, axis=0)  # Add batch dimension

# Make a prediction
predictions = loaded_model(input_data)

# Process the predictions
# You can interpret the 'predictions' based on your model's output

print(predictions)