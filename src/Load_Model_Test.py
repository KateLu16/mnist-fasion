import os
import numpy as np
import streamlit as st
import cv2

import Data_preprocessing as dataset  # type: ignore
import Activation as activation  # type: ignore
import Dense as layer  # type: ignore
import Optimization as optimization  # type: ignore
import Model as NNModel  # type: ignore
import Accuracy as accuracy  # type: ignore

# Label list
LABELS = [
    'Pull', 'Jean', 'Pull with long arm', 'Dress', 'Men Jacket',
    'High heels', 'Women shirts', 'Men shoe', 'Women handbag', 'Boots'
]

# Setup Streamlit interface
st.title("🧠 Fashion MNIST Classifier")
st.subheader("Upload a fashion image for prediction")

# Function to initialize model
@st.cache_resource
def load_trained_model(param_path: str, input_shape: int):
    model = NNModel.Model()
    model.add(layer.Layer_Dense_Regularization(input_shape, 128))
    model.add(activation.Activation_ReLU())
    model.add(layer.Layer_Dense_Regularization(128, 128))
    model.add(activation.Activation_ReLU())
    model.add(layer.Layer_Dense_Regularization(128, 10))
    model.add(activation.Activation_Linear())

    model.set(
        loss=activation.Activation_Softmax_Loss_CategoricalCrossEntropy(),
        optimizer=optimization.Optimizer_Adam(learning_rate=0.01, decay=1e-4),
        accuracy=accuracy.Accuracy_Categorical()
    )

    model.finalize()
    model.load_parameters(param_path)
    return model

# Image processing function
def preprocess_image(image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    X = np.array([image])
    X = dataset.scale(X, -1, 1)
    X = dataset.flatten(X)
    return X

# Prediction function
def predict(image_array, model):
    output = model.forward(image_array)
    prediction = np.argmax(output, axis=1)[0]
    return prediction, LABELS[prediction]

# Load model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(BASE_DIR, "..", "data_image", "MNIST_fashion_model.parms")
dummy_input = np.zeros((1, 28 * 28))
model = load_trained_model(model_file, dummy_input.shape[1])

# Upload image from user
uploaded_file = st.file_uploader("Choose PNG/JPG image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if image is not None:
        st.image(image, caption="Uploaded image", width=200)

        processed_image = preprocess_image(image)
        class_index, class_label = predict(processed_image, model)

        st.markdown(f"**Prediction:** `{class_label}` (class ID: {class_index})")
    else:
        st.error("Cannot read image. Please try again with a different image format.")
