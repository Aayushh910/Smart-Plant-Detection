from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variable to store the model
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model"""
    global model
    model_path = 'models/plant_disease_model.h5'
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("Model file not found. Please train the model first.")
        return False

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_class_names():
    """Get class names from the saved JSON file or dataset directory"""
    # Try to load from saved JSON file first
    json_path = 'models/class_names.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading class names from JSON: {e}")
    
    # Fallback to reading from dataset directory
    dataset_path = 'datasets'
    if os.path.exists(dataset_path):
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        return sorted(classes)
    return ['healthy', 'disease1', 'disease2']  # Default classes

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not available. Please train the model first.'}), 500
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Get class names
        class_names = get_class_names()
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class {predicted_class_idx}"
        
        # Get all predictions with confidence scores
        all_predictions = []
        for i, conf in enumerate(predictions[0]):
            if i < len(class_names):
                all_predictions.append({
                    'class': class_names[i],
                    'confidence': float(conf * 100)
                })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'ok',
        'model': model_status
    })

if __name__ == '__main__':
    # Try to load model on startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

