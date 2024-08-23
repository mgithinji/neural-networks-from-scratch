# building the regression model
# this model tries to get as close to a scalar value (not a classifier)

import nnfs
from nnfs.datasets import sine_data
from NeuralNetwork import (DenseLayer, Dropout, # layers
                           ReLU, Softmax, Sigmoid, Linear, # activation functions
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, # loss functions and combined activation/loss
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam, # optimizers
                           RegressionAccuracy, CategoricalAccuracy, # accuracy calculators for different types of models
                           Model) # model

# seeding random generators for dataset creation
nnfs.init()

# create dataset
X, y = sine_data()

# instantiate the model
model = Model()

# add layers to the model object
model.add(DenseLayer(1, 64))
model.add(ReLU())
model.add(DenseLayer(64, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Linear())

# set loss, optimizer, and accuracy objects
model.set(
    loss=MeanSquaredErrorLoss(),
    optimizer=Adam(learning_rate=0.005, decay=1e-3),
    accuracy=RegressionAccuracy()
)

# finalize the model
model.finalize()

# train the model
model.train(X, y, epochs=10000, print_every=1000)