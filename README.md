# 🌱 Smart Plant Health Detector

An AI-powered web application that detects plant diseases from leaf images using deep learning. Built with Flask (Python backend) and a responsive HTML/CSS/JavaScript frontend.

## 📋 Features

- **AI-Powered Detection**: Uses Convolutional Neural Network (CNN) for accurate disease classification
- **Real-time Analysis**: Get instant predictions by uploading leaf images
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Confidence Scores**: Shows prediction confidence and all possible classifications
- **User-Friendly Interface**: Modern, intuitive UI with drag-and-drop image upload
- **Extensible**: Easy to add new plant species and disease types

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   
   Create a `datasets` folder with the following structure:
   ```
   datasets/
   ├── healthy/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── disease1/
   │   ├── image1.jpg
   │   └── ...
   └── disease2/
       ├── image1.jpg
       └── ...
   ```

4. **Train the model**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load images from the `datasets` directory
   - Train a CNN model
   - Save the model to `models/plant_disease_model.h5`

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
smart/
├── app.py                 # Flask application (backend)
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── datasets/             # Training images (create this)
│   ├── healthy/
│   ├── disease1/
│   └── disease2/
├── models/               # Saved models (created after training)
│   ├── plant_disease_model.h5
│   └── class_names.json
├── uploads/              # Temporary upload storage
├── templates/            # HTML templates
│   └── index.html
└── static/              # Static files
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## 🎯 How It Works

### 1. Dataset Preparation
- Organize images by class (healthy, disease1, disease2, etc.)
- Images are automatically resized to 128x128 pixels
- Data augmentation (rotation, flipping, zoom) is applied during training

### 2. Model Training
- CNN architecture with 4 convolutional blocks
- Dropout layers to prevent overfitting
- Early stopping and model checkpointing
- Training history visualization

### 3. Disease Prediction
- User uploads a leaf image via web interface
- Image is preprocessed (resize, normalize)
- Model predicts the disease class
- Results displayed with confidence scores

## 🔧 Configuration

### Model Parameters
Edit `train_model.py` to adjust:
- `epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size for training (default: 32)
- `img_size`: Image dimensions (default: 128x128)

### Flask Settings
Edit `app.py` to change:
- `port`: Server port (default: 5000)
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 16MB)

## 📊 Model Architecture

The CNN model consists of:
- **4 Convolutional Blocks**: Feature extraction
- **Max Pooling Layers**: Dimensionality reduction
- **Dropout Layers**: Regularization (0.5)
- **Dense Layers**: Classification (512 neurons → output classes)

## 🌐 API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Upload image and get prediction
- `GET /health`: Check API and model status

## 🎨 Frontend Technologies

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS Grid and Flexbox
- **JavaScript (ES6+)**: Async/await for API calls
- **Font Awesome**: Icons
- **Responsive Design**: Mobile-first approach

## 🔮 Future Enhancements

- [ ] Treatment recommendations based on detected diseases
- [ ] Support for multiple plant species
- [ ] Batch image processing
- [ ] Historical prediction tracking
- [ ] User authentication and saved reports
- [ ] Integration with plant care databases
- [ ] Mobile app version

## 📝 Notes

- The model accuracy depends on the quality and quantity of training data
- For best results, use clear, well-lit images of leaves
- Recommended: At least 100 images per class for training
- The system is designed to be extensible - add new classes by creating new folders in `datasets/`

## 🐛 Troubleshooting

**Model not found error:**
- Make sure you've trained the model first using `train_model.py`
- Check that `models/plant_disease_model.h5` exists

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

**Upload errors:**
- Check file size (max 16MB)
- Ensure file is a valid image format (PNG, JPG, JPEG, GIF)

## 📄 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

**Built with ❤️ using Flask, TensorFlow, and modern web technologies**


