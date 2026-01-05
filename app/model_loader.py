import sys
import os

# ✅ Add SmartPlant (project root) to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from util.preprocess import preprocess_image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(os.path.join(project_root, "models", "plant_disease_model.h5"))

def predict_disease(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]

    classes = ['Tomato_Healthy', 'Tomato_Blight', 'Potato_Early_Blight', 'Potato_Late_Blight']
    return classes[class_idx]
