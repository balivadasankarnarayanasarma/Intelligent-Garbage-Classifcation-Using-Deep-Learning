from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the garbage classification model
model = load_model(r'C:\Users\baliv\OneDrive\Desktop\garbage collection\models\garbage_collection_model.h5')

# Dictionary to map class indices to class names
class_names = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        
        # Ensure the file is not empty
        if file:
            # Convert the file stream to a PIL image
            img = Image.open(file.stream)
            
            # Save the image temporarily
            temp_image_path = 'static/temp_image.png'
            img.save(temp_image_path)
            
            # Resize the image
            img = img.resize((128, 128))
            
            # Convert the PIL image to a numpy array
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            
            # Make prediction
            prediction = model.predict(img)
            
            # Get the predicted class index
            predicted_class_index = np.argmax(prediction[0])
            
            # Get the class name from the class index
            predicted_class_name = class_names[predicted_class_index]
            
            return render_template('index.html', prediction=predicted_class_name, image_path=temp_image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



