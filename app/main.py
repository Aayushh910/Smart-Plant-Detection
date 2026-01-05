from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model_loader import predict_disease
from disease_info import get_treatment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    disease = predict_disease(file_path)
    treatment = get_treatment(disease)

    return render_template('result.html', disease=disease, treatment=treatment, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
