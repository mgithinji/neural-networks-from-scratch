import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, Sigmoid, Linear, Dropout, 
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, 
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam,
                           calculate_accuracy)

nnfs.init()

PRECISION_FACTOR = 250

X, y = sine_data()

dense1 = DenseLayer(1, 64)
activation1 = ReLU()
dense2 = DenseLayer(64, 1)
activation2 = Linear()
loss_fn = MeanSquaredErrorLoss()
optimizer = Adam()

accuracy_precision = np.std(y) / PRECISION_FACTOR

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    data_loss = loss_fn.calculate(activation2.output, y)    
    regularization_loss = loss_fn.regularization_loss(dense1) + loss_fn.regularization_loss(dense2)
    
    loss = data_loss + regularization_loss
    
    predictions = activation2.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    
    if not epoch % 1000:
        print(f"epoch: {epoch}, " + 
              f"acc: {accuracy:.3f}, " + 
              f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
              f"lr: {optimizer.current_learning_rate:.3f}")

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
    
import matplotlib.pyplot as plt

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# visualizing how our model performs
plt.plot(X_test, y_test)
plt.plot(X_test, activation2.output)
plt.show()