import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# load model
model = tf.keras.models.load_model("flower_model.h5")

# class names
class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# folder containing test images
folder_path = "test_images"

# loop through all images
for filename in os.listdir(folder_path):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(folder_path, filename)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"Image: {filename}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("-------------------------")
