# building the model object

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np
from NeuralNetwork import (DenseLayer, ReLU, Softmax, Sigmoid, Linear, Dropout, 
                           CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, 
                           BinaryCrossEntropyLoss, MeanSquaredErrorLoss, MeanAbsoluteErrorLoss,
                           SGD, AdaGrad, RMSProp, Adam,
                           calculate_accuracy)

X, y = sine_data()

# input layer class
# TODO: add to the NeuralNetwork file
class InputLayer:
    # forward pass
    def forward(self, inputs):
        self.output = inputs

# model class
class Model:
    def __init__(self) -> None:
        self.layers = [] # list of network objects
        
    # add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    # set loss and optimizer
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        
    # finalize the model
    def finalize(self):
        # create and set the input layer
        self.input_layer = InputLayer()
        
        # count of hidden layers
        n_layers = len(self.layers)
        
        # initializing list of trainable layers
        self.trainable_layers = []
        
        for i in range(n_layers):
            # first layer; prev is input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # all layers except first and last
            elif i < n_layers - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # last layer; next is loss
            else: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # tracking trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
    
    # perform a forward pass
    def forward(self, X):
        self.input_layer.forward(X)
        
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        # layer is now last object from the list, return its output
        return layer.output
    
    # training the model
    def train(self, X, y, *, epochs=1, print_every=1):
        # main training in loop
        for epoch in range(1, epochs + 1):
            
            # perform the forward pass
            output = self.forward(X)
            
            print(output)
            exit()
    
    
    
# instantiate the model
model = Model()

# add layers
model.add(DenseLayer(1, 64))
model.add(ReLU())
model.add(DenseLayer(64, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Linear())

# set loss and optimizer objects
model.set(loss=MeanSquaredErrorLoss(),
          optimizer=Adam(learning_rate=0.005, decay=1e-3))

# finalize the model
model.finalize()

# training the model
model.train(X, y, epochs=10000, print_every=1000)

print(model.layers)