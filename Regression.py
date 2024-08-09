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
dense2 = DenseLayer(64, 64)
activation2 = ReLU()
dense3 = DenseLayer(64, 1)
activation3 = Linear()
loss_fn = MeanSquaredErrorLoss()
optimizer = Adam(learning_rate=0.005, decay=1e-3)

accuracy_precision = np.std(y) / PRECISION_FACTOR

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    data_loss = loss_fn.calculate(activation3.output, y)    
    regularization_loss = loss_fn.regularization_loss(dense1) + \
                          loss_fn.regularization_loss(dense2) + \
                          loss_fn.regularization_loss(dense3)
    
    loss = data_loss + regularization_loss
    
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    
    if not epoch % 1000:
        print(f"epoch: {epoch}, " + 
              f"acc: {accuracy:.3f}, " + 
              f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
              f"lr: {optimizer.current_learning_rate:.5f}")

    # backward pass
    loss_fn.backward(activation3.output, y)
    activation3.backward(loss_fn.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
    
import matplotlib.pyplot as plt

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

# visualizing how our model performs
plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()