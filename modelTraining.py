import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns


class_names = {0:"Bomber", 1:"Fighters", 2:"Fire", 3:"Firefighting", 4:"Surveillance", 5:"ThingsToNotShoot", 6:"Transport", 7:"Unmanned", 8:"Vistas"}

imgs_path = "D:\Coding\School\img_rec_proj\Output\\train"
data = []
labels = []
classes = 9
for i in range(classes):
    img_path = os.path.join(imgs_path + "\\", list(class_names.values())[i]) #0-9
    for img in os.listdir(img_path):
        try:
            print(img_path + '\\' + img)
            im = Image.open(img_path + '\\' + img)
            im = im.resize((224,224,3)[:2])
            im = np.array(im)
            # Check if the image has 3 channels (RGB) and resize if necessary
            # Ensure that the image has 3 channels (RGB)
            if im.shape[-1] != 3:
                raise ValueError("Image does not have 3 channels (RGB)")
            data.append(im)
            labels.append(list(class_names.values())[i])
        except:
            print("FUCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCK | " + img_path + '\\' + img)
            continue
            

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=object)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
#splitthedata
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, train_size=0.8, random_state=random.randint(4,64))
print("training shape: ",x_train.shape, y_train.shape)
print("testing shape: ",x_test.shape, y_test.shape)
y_train = to_categorical(y_train, 9)
y_test = to_categorical(y_test, 9)

# Convert one-hot encoded labels back to integer labels
y_train_labels = np.argmax(y_train, axis=1)

print("data loaded!")

#model = tf.keras.models.load_model('military_aircraft_classifier.h5')
model = tf.keras.models.load_model(filepath="D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5")

# Recompile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[keras.metrics.F1Score(threshold=0.5),keras.metrics.AUC()])

# Initialize a list to store training histories
all_histories = []

for _ in range(1):
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
    tf.keras.models.save_model(model=model, filepath="D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5", save_format='h5')
    print("SAVED!!!!")
    all_histories.append(history)

print("model modelled")
# Initialize lists to store training and validation accuracy/loss across sessions
train_acc = []
val_acc = []
train_loss = []
val_loss = []

# Collect the accuracy and loss data across sessions
for history in all_histories:
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])
    train_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])


# Plot training and validation accuracy on a single plot

plt.figure(0)
plt.plot(train_acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot training and validation loss on a single plot
plt.figure(1)
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Show all the plots
plt.show()

# Assuming your model is named 'model', make predictions on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded test labels back to integer labels
y_test_labels = np.argmax(y_test, axis=1)

# Generate the confusion matrix
confusion_mtx = confusion_matrix(y_test_labels, y_pred_classes)

# Plot the confusion matrix
plt.figure(2)
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()