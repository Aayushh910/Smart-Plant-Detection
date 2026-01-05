import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset folder
dataset_path = r"C:\SmartPlant\datasets"

# Define image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Create ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Display total images
print(f"✅ Training images: {train_generator.samples}")
print(f"✅ Validation images: {val_generator.samples}")

# Save class names for future reference
class_labels = list(train_generator.class_indices.keys())
print(f"🌿 Detected Classes: {class_labels}")

with open("class_labels.txt", "w") as f:
    for label in class_labels:
        f.write(label + "\n")
