# testing the optimizer

from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, 
                           CategoricalCrossEntropyLoss, 
                           SoftmaxActivationCCELoss, SGD, calculate_accuracy)

nnfs.init()

# data
X, y = spiral_data(samples=100, classes=3)

print(X[:5])
print()

# network
dense1 = DenseLayer(2, 64)
activation1 = ReLU()
dense2 = DenseLayer(64, 3)
loss_activation = SoftmaxActivationCCELoss()
optimizer = SGD(decay=1e-2)

# training in a loop
for epoch in range(10001):
    
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    accuracy = calculate_accuracy(loss_activation.output, y)

    if not epoch % 1000:
        print("epoch: {}, acc: {}, loss: {}, lr: {}".format(epoch, accuracy, loss, 
                                                            optimizer.current_learning_rate))

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()