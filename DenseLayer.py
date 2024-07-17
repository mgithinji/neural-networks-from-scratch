# coding a dense layer (or fully connected layer) class

from nnfs.datasets import spiral_data
import nnfs
import numpy as np

nnfs.init()

class DenseLayer:
    # layer initialization
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3)
     
dense1 = DenseLayer(n_inputs=2, n_neurons=3)
dense1.forward(X)

print(dense1.output[:5])