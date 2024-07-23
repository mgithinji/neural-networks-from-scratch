# coding the backpropagation method for a layer of neurons
# 3 neurons receiving 4 inputs (3 weight sets, 4 weights each)

import numpy as np

# arbitrarily chosen derivative values from the "next" layer
dvalues = np.array([[1.0, 1.0, 1.0]]) # vector of 1s

weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# sum weights related to the given input multiplied by the gradient related to given input

dx0 = sum([weights[0][0] * dvalues[0][0], weights[0][1] * dvalues[0][1], weights[0][2] * dvalues[0][2]])
dx1 = sum([weights[1][0] * dvalues[0][0], weights[1][1] * dvalues[0][1], weights[1][2] * dvalues[0][2]])
dx2 = sum([weights[2][0] * dvalues[0][0], weights[2][1] * dvalues[0][1], weights[2][2] * dvalues[0][2]])
dx3 = sum([weights[3][0] * dvalues[0][0], weights[3][1] * dvalues[0][1], weights[3][2] * dvalues[0][2]])
dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# same result as above, because the weight array is formatted such that rows contain weights related to each input
dx0 = sum(weights[0] * dvalues[0])
dx1 = sum(weights[1] * dvalues[0])
dx2 = sum(weights[2] * dvalues[0])
dx3 = sum(weights[3] * dvalues[0])
dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# same result as above, using dot product
dinputs = np.dot(dvalues[0], weights.T)
print(dinputs)

# trying this with a batch of samples

# batch of samples
dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])

dinputs = np.dot(dvalues, weights.T)
print(dinputs)

# 3 sets of inputs (4 inputs each)
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

# calculating gradient wrt weights
# since the derivative wrt weights = inputs, and weights are transposed, we transpose the inputs here
dweights = np.dot(inputs.T, dvalues)
print(dweights)

# 3 biases for our 3 neurons
# row vector with a shape (1, n_neurons)
biases = np.array([[2.0, 3.0, 0.5]])

# calculating gradient wrt biases
# since the gradients are a list of gradients, we just have to sum them with the neurons, column wise (axis=0)
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)