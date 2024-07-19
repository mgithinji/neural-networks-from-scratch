# experimenting with optimization techniques

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data, spiral_data
from NeuralNetwork import DenseLayer, ReLU, Softmax, CategoricalCrossEntropyLoss, calculate_accuracy

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

dense1 = DenseLayer(n_inputs=2, n_neurons=3)
activation1 = ReLU()
dense2 = DenseLayer(n_inputs=3, n_neurons=3)
activation2 = Softmax()

loss_fn = CategoricalCrossEntropyLoss()

# helper variables, initial values
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(100000):
    
    # generate new set of random weight and bias values for iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)
    
    # perform forward pass of training data through layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # perform forward pass through loss function
    loss = loss_fn.calculate(activation2.output, y)
    accuracy = calculate_accuracy(activation2.output, y)
    
    if loss < lowest_loss:
        print("{}: new parameters found -- loss: {}, acc: {}".format(i, loss, accuracy))
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    else: # revert to previous parameters
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()