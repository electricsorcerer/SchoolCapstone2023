import tensorflow as tf

model = tf.keras.models.load_model('D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5')
tf.saved_model.save(model, 'D:\Coding\School\img_rec_proj\TheModel')