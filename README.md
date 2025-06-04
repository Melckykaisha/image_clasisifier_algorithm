# 🧠 CIFAR-10 Image Classifier Web App

This is a simple web application that allows users to upload an image and get a prediction from a trained image classification model based on the CIFAR-10 dataset.

## 🚀 Features

- Upload an image (`.jpg`, `.jpeg`, or `.png`)
- Automatically preprocesses and resizes it to 32x32
- Uses a pretrained CNN model to classify the image into one of 10 categories
- Displays the predicted class and confidence score

## 🖼 CIFAR-10 Classes

The model classifies images into the following 10 categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## 🧰 Technologies Used

- **TensorFlow/Keras** – for training and loading the CNN model  
- **Streamlit** – to create the interactive web app  
- **Python** – for backend logic  
- **NumPy & PIL** – for image preprocessing
- **HTML,CSS and flask** - for web based

## 📦 Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit tensorflow pillow numpy
