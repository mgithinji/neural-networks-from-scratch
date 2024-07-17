# coding a two-layer neural network

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# weights and biases for neurons in layer 1
weights1 = [[0.2, 0.8, -0.5, 1.0], # weights for neuron(1,1)
            [0.5, -0.91, 0.26, -0.5], # weights for neuron(1,2)
            [-0.26, -0.27, 0.17, 0.87]] # weights for neuron(1,3)

biases1 = [2, 3, 0.5]

# weights and biases for neurons in layer 2
weights2 = [[0.1, -0.14, 0.5], # weights for neuron(2,1)
            [-0.5, 0.12, -0.33], # weights for neuron(2,2)
            [-0.44, 0.73, -0.13]] # weights for neuron(2,3)

biases2 = [-1.0, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)