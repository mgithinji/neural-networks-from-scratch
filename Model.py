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
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        
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
                
            # tracking trainable layers -- those that have a weight attribute
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
        # updating loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
    
    # perform a forward pass
    def forward(self, X):
        self.input_layer.forward(X)
        
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        # "layer" is now last object from the list, return its output
        return layer.output
    
    # perform a backward pass
    def backward(self, output, y):
        # start by calling backward method on the loss to set dinputs for the last layer to use
        self.loss.backward(output, y)
        
        # call backward method going through all network objects in reverse
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    # training the model
    def train(self, X, y, *, epochs=1, print_every=1):
        # initialize accuracy object
        self.accuracy.init(y)
        
        # main training in loop
        for epoch in range(1, epochs + 1):
            
            # perform the forward pass
            output = self.forward(X)
            
            # calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss
            
            # get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            
            # perform backward pass
            self.backward(output, y)
            
            # optimize / update params
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            
            # print training summary
            if not epoch % print_every:
                print(f"epoch: {epoch}, " + 
                      f"acc: {accuracy:.3f}, " + 
                      f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
                      f"lr: {self.optimizer.current_learning_rate:.5f}")
            
            
    
# creating accuracy classes

# base accuracy class
class Accuracy:
    # calculating accuracy
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy
    
# accuracy for regression model
class RegressionAccuracy(Accuracy):
    def __init__(self) -> None:
        self.precision = None
    
    # calculating the precision value
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    # comparing predictions and ground truth
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    
    
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
          optimizer=Adam(learning_rate=0.005, decay=1e-3),
          accuracy=RegressionAccuracy())

# finalize the model
model.finalize()

# training the model
model.train(X, y, epochs=10000, print_every=1000)