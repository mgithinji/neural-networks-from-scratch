import numpy as np

# coding a layer of neurons with batch input

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0], # weights for neuron1
            [0.5, -0.91, 0.26, -0.5], # weights for neuron2
            [-0.26, -0.27, 0.17, 0.87]] # weights for neuron3

biases = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)