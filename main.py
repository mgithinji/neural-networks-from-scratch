# coding the neural network with the classes and methods created

from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, Dropout, 
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, 
                           SGD, AdaGrad, RMSProp, Adam,
                           calculate_accuracy)

nnfs.init()

# data
X, y = spiral_data(samples=1000, classes=3)

# network
dense1 = DenseLayer(2, 64, lambda_l2w=5e-4, lambda_l2b=5e-4)
activation1 = ReLU()
dropout1 = Dropout(rate=0.1)
dense2 = DenseLayer(64, 3)
loss_activation = SoftmaxActivationCCELoss()
optimizer = Adam(learning_rate=0.05, decay=5e-5)

# training in a loop
for epoch in range(10001):
    
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    accuracy = calculate_accuracy(loss_activation.output, y)

    if not epoch % 1000:
        print(f"epoch: {epoch}, " + 
              f"acc: {accuracy:.3f}, " + 
              f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
              f"lr: {optimizer.current_learning_rate:.3f}")

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
# validate the model
X_test, y_test = spiral_data(samples=100, classes=3)

# forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
accuracy = calculate_accuracy(loss_activation.output, y_test)

print("Validation - acc: {:.3f}, loss: {:.3f}".format(accuracy, loss))