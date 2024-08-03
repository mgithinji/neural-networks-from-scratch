# coding the neural network with the classes and methods created

from nnfs.datasets import spiral_data
import nnfs
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, Sigmoid, Dropout, 
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, BinaryCrossEntropyLoss,
                           SGD, AdaGrad, RMSProp, Adam,
                           calculate_accuracy)

nnfs.init()

# testing data
X, y = spiral_data(samples=100, classes=2)

# reshaping our labels to work with a binary logistic regression classifier - labels are currently sparse
y = y.reshape(-1, 1)

# network
dense1 = DenseLayer(2, 64, lambda_l2w=5e-4, lambda_l2b=5e-4)
activation1 = ReLU()
dense2 = DenseLayer(64, 1)
activation2 = Sigmoid()
loss_fn = BinaryCrossEntropyLoss()
optimizer = Adam(decay=5e-7)

# training in a loop
for epoch in range(10001):
    
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_fn.calculate(activation2.output, y)
    regularization_loss = loss_fn.regularization_loss(dense1) + loss_fn.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if not epoch % 1000:
        print(f"epoch: {epoch}, " + 
              f"acc: {accuracy:.3f}, " + 
              f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
              f"lr: {optimizer.current_learning_rate:.3f}")

    # backward pass
    loss_fn.backward(activation2.output, y)
    activation2.backward(loss_fn.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
# validate the model

# test data
X_test, y_test = spiral_data(samples=100, classes=2)

# reshape labels to fit a binary logistic regression architecture
y_test = y_test.reshape(-1, 1)

# forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_fn.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print("Validation - acc: {:.3f}, loss: {:.3f}".format(accuracy, loss))