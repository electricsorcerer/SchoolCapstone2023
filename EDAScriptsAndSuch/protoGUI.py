import cv2
import numpy as np
import tensorflow as tf
import skimage
import tkinter as tk
from tkinter import filedialog
import tkinter as tk
from tkinter import filedialog
from tkinter import Text
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the trained model for image classification
classification_model = tf.keras.models.load_model('D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5')

# Function to open an image file and process it
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        im = Image.open(file_path)
        im = im.resize((224,224,3)[:2])
        im = np.array(im)
        # Check if the image is in the correct format (CV_8U) before converting
        predictions = classification_model.predict(np.expand_dims(im, axis=0))
        class_index = np.argmax(predictions)
        print(class_index)
        print(predictions)
        confidence = predictions[0, class_index]
        print(f"{confidence}, {class_names[class_index]}")
        cv2.imshow("Processed Image", im)
        cv2.waitKey(0)

# Function to open a video file and process it
def open_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        video_capture = cv2.VideoCapture(file_path)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))  # Correctly resize the frame
            frame = np.array(frame)
            predictions = classification_model.predict(np.expand_dims(frame, axis=0))
            class_index = np.argmax(predictions)
            print(class_index)
            print(predictions)
            confidence = predictions[0, class_index]
            print(f"{confidence}, {class_names[class_index]}")
            cv2.imshow("Processed Image", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Create the main GUI window
root = tk.Tk()
root.title("Image and Video Processing")

# Create buttons for opening images and videos
image_button = tk.Button(root, text="Open Image", command=open_image)
image_button.pack()
video_button = tk.Button(root, text="Open Video", command=open_video)
video_button.pack()

# Replace 'classification_model.h5' with the actual path to your image classification model
class_names = {0:"Bomber", 1:"Fighters", 2:"Fire", 3:"Firefighting", 4:"Surveillance",
5:"ThingsToNotShoot", 6:"Transport", 7:"Unmanned", 8:"Vistas"}

root.mainloop()
