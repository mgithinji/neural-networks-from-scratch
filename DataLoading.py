# reading image data and associating images with labels
import os
import cv2
import numpy as np

def load_mnist_dataset(dataset, path):
    # scanning all directories and creating a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # creating lists for samples and labels
    X = []
    y = []

    # moving samples and labels into our lists
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
            
    return np.array(X), np.array(y).astype('uint8')

def create_mnist_dataset(path):
    # load both datasets separately
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    
    # return all
    return X, y, X_test, y_test