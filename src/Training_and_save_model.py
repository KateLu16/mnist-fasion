import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import Data_preprocessing as dataset  # type: ignore
import Activation as activation  # type: ignore
import Dense as layer  # type: ignore
import Optimization as optimization  # type: ignore
import Model as NNModel  # type: ignore
import Loss as loss_function  # type: ignore
import Accuracy as accuracy  # type: ignore

def load_and_preprocess_data(data_path):
    print("1. Loading and preprocessing the dataset...")
    X, y, X_test, y_test = dataset.create_data_MNIST(data_path)

    # Scale the data to [-1, 1]
    X = dataset.scale(X, -1, 1)
    X_test = dataset.scale(X_test, -1, 1)

    # Flatten images to 1D vectors
    X = X.reshape((X.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Shuffle training data
    train_keys = np.arange(X.shape[0])
    np.random.shuffle(train_keys)
    X = X[train_keys]
    y = y[train_keys]

    # Shuffle test data
    test_keys = np.arange(X_test.shape[0])
    np.random.shuffle(test_keys)
    X_test = X_test[test_keys]
    y_test = y_test[test_keys]

    print("   → Dataset ready!")
    return X, y, X_test, y_test

def build_model(input_dim):
    print("2. Building the model...")
    model = NNModel.Model()
    model.add(layer.Layer_Dense_Regularization(input_dim, 128))
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
    print("   → Model is ready!")
    return model

def train_and_save_model(model, X, y, X_val, y_val, save_path):
    print("3. Start training...")
    model.train(X, y, validation_data=(X_val, y_val), epochs=100, print_every=10)

    print("\n4. Saving model to:", save_path)
    model.save_parameters(save_path)
    print("   → Model saved successfully!") 

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data_image")
    save_path = os.path.join(data_path, "MNIST_fashion_model.parms")

    X, y, X_test, y_test = load_and_preprocess_data(data_path)
    model = build_model(input_dim=X.shape[1])
    train_and_save_model(model, X, y, X_test, y_test, save_path)

if __name__ == "__main__":
    main()

