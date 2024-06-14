from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import gdown

app = Flask(__name__)

url = "https://drive.google.com/uc?id=1VaMb2oy8zYOsa7aq1x6MzxjOpPANHZk5"
output = 'model.h5'  # مسار الحفظ المحلي

# تنزيل الملف من Google Drive
gdown.download(url, output, quiet=False)

# تحميل النموذج باستخدام TensorFlow/Keras
model = load_model(output)


# Define a function to preprocess input images
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust based on your model input size
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define your classes
        classes = ['Coccidiosis', 'Healthy', 'Salmonilla']

        return jsonify({'class': classes[predicted_class], 'probability': float(np.max(predictions))})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
