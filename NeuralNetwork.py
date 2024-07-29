# code for neural network functions

import numpy as np
from abc import ABC, abstractmethod

class DenseLayer:
    # layer initialization
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases        
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# abstract base class for activation functions
class Activation(ABC):
    # forward pass
    @abstractmethod
    def forward(self):
        pass

# ReLU activation function
class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation funtion
class Softmax(Activation):
    def forward(self, inputs):
        # calculating raw probabilities
        # NOTE: we subtract the largest inputs before calc to mitigate "exploding values"
        prob_raw = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob_norm = prob_raw = prob_raw / np.sum(prob_raw, axis=1, keepdims=True)
        self.output = prob_norm
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) # uninitialized array
        
        for i, (i_output, i_dvalues) in enumerate(zip(self.output, dvalues)):
            # creating the Jacobian matrix of the partial derivatives of the softmax function
            i_output = i_output.reshape(-1, 1)
            jacobian = np.diagflat(i_output) - np.dot(i_output, i_output.T)
            
            # calculating the sample-wise gradient and adding it to the sample gradients
            self.dinputs[i] = np.dot(jacobian, i_dvalues)
            
        
# base Loss class
class Loss(ABC):
    # forward pass
    @abstractmethod
    def forward(self):
        pass
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# categortical cross entropy loss class, inheriting from base loss class
class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred) # num samples in batch
        
        # clip data on both sides to prevent division by 0 and shifting
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # adding a condition for one-hot encoded vs sparse target inputs
        if len(y_true.shape) == 1: # sparse
            target_confidence_scores = y_pred_clipped[range(n_samples), y_true]
        elif len(y_true.shape) == 2: # one-hot encoded
            target_confidence_scores = np.sum(y_pred_clipped * y_true, 
                                              axis=1)
        
        negative_log_likelihoods = -np.log(target_confidence_scores)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues) # number of samples
        n_labels = len(dvalues[0]) # number of labels in each sample
        
        # adding a condition to turn sparse targets into one-hot encoded vector
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
            
        self.dinputs = -y_true / dvalues # calculate gradient
        self.dinputs = self.dinputs / n_samples # normalize gradient

# combined Softmax and CCE loss class
class SoftmaxActivationCCELoss():
    def __init__(self) -> None:
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyLoss()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs=inputs)
        self.output = self.activation.output
        return self.loss.calculate(output=self.activation.output, y=y_true)
    
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues) # number of samples
        
        # if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), y_true] -= 1 # calculating gradient
        self.dinputs = self.dinputs / n_samples # normalizing gradient
        
# base optimizer class
class Optimizer(ABC):
    # update parameters
    @abstractmethod
    def update_params(self):
        pass

# stochastic gradient descent optimizer
class SGD(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        
    # to be called once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        
        # when using momentum
        if self.momentum:
            # if layer doesn't have a momentum array, create one
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # build weight updates with momentum - previous updates * retain factor and update with current gradients
            weight_updates = (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates
            
            # build bias updates with momentum
            bias_updates = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates
        
        # without momentum
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # update our actual weights and biases  
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    # to be called after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
# adaptive gradient (AdaGrad) descent optimizer
class AdaGrad(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        
    # to be called once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        
        # if layer doesn't have cache arrays, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        # parameter update and normalization w/ sqrt cache
        layer.weights += (-self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += (-self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    # to be called after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
# root mean square propagation (RMSProp) descent optimizer
class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0
        
    # to be called once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        
        # if layer doesn't have cache arrays, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # update cache with squared current gradients
        layer.weight_cache = (self.rho * layer.weight_cache) + ((1 - self.rho) * (layer.dweights ** 2))
        layer.bias_cache = (self.rho * layer.bias_cache) + ((1 - self.rho) * (layer.dbiases ** 2))
        
        # parameter update and normalization w/ sqrt cache
        layer.weights += (-self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += (-self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    # to be called after any parameter updates
    def post_update_params(self):
        self.iterations += 1

def calculate_accuracy(output, y):
    # get the predictions in a single vector
    predictions = np.argmax(output, axis=1)
    
    # convert one-hot encoded target inputs to a single vector
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    accuracy = np.mean(predictions == y)
    return accuracy