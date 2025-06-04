from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("cifar10_model.h5")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    file = request.files['image']
    if file:
        image = Image.open(file).resize((32, 32))
        img_array = np.array(image) / 255.0
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]  # remove alpha
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', label=predicted_class)

    return 'No file uploaded.', 400

if __name__ == '__main__':
    app.run(debug=True)
