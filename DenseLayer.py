# coding a dense layer (or fully connected layer) class

from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from abc import ABC, abstractmethod

nnfs.init()

class DenseLayer:
    # layer initialization
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# abstract base class for activation functions
class Activation(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    # forward pass
    @abstractmethod
    def forward(self):
        pass

# ReLU activation function
class ReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation funtion
class Softmax(Activation):
    def forward(self, inputs):
        # calculating raw probabilities
        # NOTE: we subtract the largest inputs before calc to mitigate "exploding values"
        prob_raw = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob_norm = prob_raw = prob_raw / np.sum(prob_raw, axis=1, keepdims=True)
        self.output = prob_norm

X, y = spiral_data(samples=100, classes=3)
     
dense1 = DenseLayer(n_inputs=2, n_neurons=3)
activation1 = ReLU()
dense2 = DenseLayer(n_inputs=3, n_neurons=3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

softmax1 = Softmax()
softmax1.forward([[1,2,3]])
print(softmax1.output)