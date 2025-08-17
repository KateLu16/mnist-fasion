import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_MNIST_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    
    # Create lists for samples and labels
    X = []
    y = []
    
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file),
                               cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')

def create_data_MNIST(path):
    # Load both sets separately
    X, y = load_MNIST_dataset('train', path)
    X_test, y_test = load_MNIST_dataset('test', path)
    return X, y, X_test, y_test

# Scale range from 0-255 to -1 to 1
def scale(X, min_val, max_val):
    X_scaled = (X.astype(np.float32) - 127.5) / 127.5
    return X_scaled

def flatten(X):
    X_reshaped = X.reshape((X.shape[0], -1))
    return X_reshaped

def shuffle(X, y):
    # Create keys to allow shuffling 2 vectors X and y simultaneously
    keys = np.arange(X.shape[0])
    np.random.shuffle(keys)
    X_shuffled = X[keys]
    y_shuffled = y[keys]
    return X_shuffled, y_shuffled
