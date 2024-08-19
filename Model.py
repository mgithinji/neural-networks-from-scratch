# building the model object

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data, spiral_data
import numpy as np
from NeuralNetwork import (DenseLayer, Dropout, # layers
                           ReLU, Softmax, Sigmoid, Linear, # activation functions
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, # loss functions and combined activation/loss
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam, # optimizers
                           RegressionAccuracy, CategoricalAccuracy, # accuracy calculators for different types of models
                           Model) # model


# creating training and test dataset
X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# instantiate the model
model = Model()

# add layers
model.add(DenseLayer(2, 512, lambda_l2w=5e-4, lambda_l2b=5e-4))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(DenseLayer(512, 3))
model.add(Softmax())

# set loss and optimizer objects
model.set(loss=CategoricalCrossEntropyLoss(),
          optimizer=Adam(learning_rate=0.05, decay=5e-5),
          accuracy=CategoricalAccuracy())

# finalize the model
model.finalize()

# training the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=1000)