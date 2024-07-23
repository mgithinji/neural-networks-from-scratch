# coding the backpropagation method for a layer of neurons
# 3 neurons receiving 4 inputs (3 weight sets, 4 weights each)
# and ReLU activation

import numpy as np

# passed in gradient from next layer
dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])

# 3 sets of inputs (4 inputs each)
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

# 3 sets of weights, one for each neuron
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# 3 biases for our 3 neurons; row vector with a shape (1, n_neurons)
biases = np.array([[2.0, 3.0, 0.5]])

# forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# optimizing and testing backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# dense layer

dinputs = np.dot(drelu, weights.T) # multiply by weights 
dweights = np.dot(inputs.T, drelu) # multiply by inputs
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)