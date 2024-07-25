# coding the network from our created classes

from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, 
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, calculate_accuracy)

nnfs.init()

# building a network from our classes

X, y = spiral_data(samples=100, classes=3)
     
dense1 = DenseLayer(n_inputs=2, n_neurons=3)
activation1 = ReLU()
dense2 = DenseLayer(n_inputs=3, n_neurons=3)
loss_activation = SoftmaxActivationCCELoss() # combined softmax and loss fn

# forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
accuracy = calculate_accuracy(output=loss_activation.output, y=y)

print(loss_activation.output[:5]) # output of the first few samples
print("loss: {}".format(loss)) # loss value
print("accuracy: {}".format(accuracy)) # accuracy value

# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)