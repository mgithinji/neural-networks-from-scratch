# building a binary logistic regression model
# this model tries to classify data into a single class (T/F, Yes/No, Cat/Dog, 0/1, etc)

import nnfs
from nnfs.datasets import spiral_data
from NeuralNetwork import (DenseLayer, Dropout, # layers
                           ReLU, Softmax, Sigmoid, Linear, # activation functions
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, # loss functions and combined activation/loss
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam, # optimizers
                           RegressionAccuracy, CategoricalAccuracy, # accuracy calculators for different types of models
                           Model) # model

# seeding random generators for dataset creation
nnfs.init()

# creating train and test dataset
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# reshaing labels to be a list of lists, instead of parse labels
# matchin the format our model object expects
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# instantiate the model
model = Model()

# add layers to the model
model.add(DenseLayer(2, 64, lambda_l2w=5e-4, lambda_l2b=5e-4))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Sigmoid())

# set loss, optimizer, and accuracy objects
model.set(
    loss=BinaryCrossEntropyLoss(),
    optimizer=Adam(decay=5e-7),
    accuracy=CategoricalAccuracy(binary=True)
)

# finalize the model
model.finalize()

# train the model
model.train(X, y, 
            validation_data=(X_test, y_test), 
            epochs=10000, print_every=1000)