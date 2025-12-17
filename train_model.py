"""
Train CNN model for plant disease detection
"""
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape=(128, 128, 3), num_classes=None):
    """
    Create a CNN model for plant disease classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of disease classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset(dataset_path='datasets', img_size=(128, 128)):
    """
    Load images from dataset directory
    
    Args:
        dataset_path: Path to dataset directory
        img_size: Target image size (height, width)
    
    Returns:
        X: Array of images
        y: Array of labels
        class_names: List of class names
    """
    X = []
    y = []
    class_names = []
    
    if not os.path.exists(dataset_path):
        print(f"Dataset directory '{dataset_path}' not found!")
        print("Please create the dataset directory with the following structure:")
        print("datasets/")
        print("  ├── healthy/")
        print("  ├── disease1/")
        print("  └── disease2/")
        return None, None, None
    
    # Get all class directories
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')])
    
    if len(classes) == 0:
        print(f"No class directories found in '{dataset_path}'!")
        return None, None, None
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Load images from each class
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        print(f"Loading {len(images)} images from '{class_name}'...")
        
        for img_name in images:
            try:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img)
                img_array = img_array.astype('float32') / 255.0
                
                X.append(img_array)
                y.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to categorical
    y_categorical = keras.utils.to_categorical(y, num_classes=len(classes))
    
    print(f"\nDataset loaded successfully!")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Number of classes: {len(classes)}")
    
    return X, y_categorical, classes

def train_model(epochs=20, batch_size=32, validation_split=0.2):
    """
    Train the CNN model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    """
    print("=" * 50)
    print("Plant Disease Detection Model Training")
    print("=" * 50)
    
    # Load dataset
    X, y, class_names = load_dataset()
    
    if X is None:
        print("\nTraining aborted. Please set up your dataset first.")
        return
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(input_shape=X[0].shape, num_classes=len(class_names))
    model.summary()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'models/plant_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    print("=" * 50)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 50)
    print("Evaluating model...")
    # Load best model if checkpoint exists (ModelCheckpoint saves the best model)
    if os.path.exists('models/plant_disease_model.h5'):
        model = keras.models.load_model('models/plant_disease_model.h5')
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Save class names
    os.makedirs('models', exist_ok=True)
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Model saved to: models/plant_disease_model.h5")
    print("=" * 50)

def plot_training_history(history):
    """Plot training history"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("Training history plot saved to: models/training_history.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {e}")

if __name__ == '__main__':
    train_model(epochs=20, batch_size=32)

