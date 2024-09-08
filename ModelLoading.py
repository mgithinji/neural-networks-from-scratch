# code for creating a model with a real dataset (Fashion MNIST)
import numpy as np
import nnfs
from NeuralNetwork import (DenseLayer, Dropout, # layers
                           ReLU, Softmax, Sigmoid, Linear, # activation functions
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, # loss functions and combined activation/loss
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam, # optimizers
                           RegressionAccuracy, CategoricalAccuracy, # accuracy calculators for different types of models
                           Model) # model
from DataLoading import create_mnist_dataset

nnfs.init()

# creating the dataset
X, y, X_test, y_test = create_mnist_dataset("fashion_mnist_images")

# scaling image data to have the range (-1, 1)
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# reshaping image data to a vector for our neural network
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# shuffling our image data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# instantiate the model
model = Model.load("fashion_mnist.model")

model.evaluate(X_test, y_test)