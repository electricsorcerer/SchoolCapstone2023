import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time 

# Load the trained model for image classification
classification_model = tf.keras.models.load_model('D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5')

# Replace 'classification_model.h5' with the actual path to your image classification model
class_names = {0: "Bomber", 1: "Fighters", 2: "Fire", 3: "Firefighting", 4: "Surveillance",
              5: "ThingsToNotShoot", 6: "Transport", 7: "Unmanned", 8: "Vistas"}

# Create the main GUI window
root = tk.Tk()
root.title("Image and Video Processing")

# Create labels for displaying image and confidence
image_label = tk.Label(root)
image_label.pack()
confidence_label = tk.Label(root)
confidence_label.pack()

# Function to open an image file and process it
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        im = Image.open(file_path)
        im = im.resize((224, 224))
        im = np.array(im)

        predictions = classification_model.predict(np.expand_dims(im, axis=0))
        class_index = np.argmax(predictions)
        confidence = predictions[0, class_index]

        # Update the labels to display the image and confidence
        img = ImageTk.PhotoImage(image=Image.fromarray(im))
        image_label.configure(image=img)
        image_label.image = img

        confidence_label.config(text=f"Class: {class_names[class_index]}, Confidence: {confidence:.2f}")

# Function to open a video file and process it
def open_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        video_capture = cv2.VideoCapture(file_path)

        frame_count = 0
        start_time = time.time()

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1

            frame = cv2.resize(frame, (224, 224))
            frame = np.array(frame)
            predictions = classification_model.predict(np.expand_dims(frame, axis=0))
            class_index = np.argmax(predictions)
            confidence = predictions[0, class_index]

            # Add an FPS counter on the frame
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Processed Image", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Create buttons for opening images and videos
image_button = tk.Button(root, text="Open Image", command=open_image)
image_button.pack()
video_button = tk.Button(root, text="Open Video", command=open_video)
video_button.pack()

root.mainloop()