# code for neural network functions

import numpy as np
from abc import ABC, abstractmethod
import pickle
import copy

# input layer class
class InputLayer:
    # forward pass
    def forward(self, inputs, training):
        self.output = inputs

class DenseLayer:
    # layer initialization
    def __init__(self, n_inputs: int, n_neurons: int,
                 lambda_l1w=0, lambda_l2w=0, 
                 lambda_l1b=0, lambda_l2b=0) -> None:
        # initializing weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
        # setting regularization strength
        self.lambda_l1w = lambda_l1w
        self.lambda_l2w = lambda_l2w
        self.lambda_l1b = lambda_l1b
        self.lambda_l2b = lambda_l2b
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases        
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # gradients on regularization
        # L1 regularization on weights
        if self.lambda_l1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.lambda_l1w * dL1
        
        # L2 regularization on weights   
        if self.lambda_l2w > 0:
            self.dweights += 2 * self.lambda_l2w * self.weights
            
        # L1 regularization on biases
        if self.lambda_l1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.lambda_l1b * dL1
            
        # L2 regularization on biases
        if self.lambda_l2b > 0:
            self.dbiases += 2 * self.lambda_l2b * self.biases
            
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
    # retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # set layer parameters
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# creating the Dropout layer class
class Dropout:
    def __init__(self, rate) -> None:
        self.rate = 1 - rate
        
    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs # save input values
        
        if not training:
            self.output = inputs.copy()
            return
        
        self.binary_mask = np.random.binomial(n=1, p=self.rate, size=inputs.shape) / self.rate        
        self.output = inputs * self.binary_mask # apply mask to input values
        
    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

# abstract base class for activation functions
class Activation(ABC):
    # forward pass
    @abstractmethod
    def forward(self):
        pass
    
    # backward pass
    @abstractmethod
    def backward(self):
        pass
    
    # returning predictions
    @abstractmethod
    def predictions(self):
        pass

# ReLU activation function
class ReLU(Activation):
    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    # calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax activation funtion
class Softmax(Activation):
    def forward(self, inputs, training):
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
    
    # calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# sigmoid activation - commonly used in binary logistic regression
class Sigmoid(Activation):
    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs # saving inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
    # calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# linear activation - commonly used in regression
class Linear(Activation):
    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs # saving values
        self.output = inputs
    
    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    
    # calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# base Loss class
class Loss(ABC):
    # forward pass
    @abstractmethod
    def forward(self):
        pass
    
    # backward pass
    @abstractmethod
    def backward(self):
        pass
    
    # set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, * , include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        
        # add accumulated sum of losses and sample count (for batch training)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    # calculated accumulated loss when training in batches
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    # reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def regularization_loss(self):
        # regularization loss is 0 by default
        regularization_loss = 0
        
        # calculate regularization loss by iterating over all trainable layers
        for layer in self.trainable_layers:
        
            # L1 regularization - weights
            if layer.lambda_l1w > 0:
                regularization_loss += layer.lambda_l1w * np.sum(np.abs(layer.weights))
                
            # L2 regularization - weights
            if layer.lambda_l2w > 0:
                regularization_loss += layer.lambda_l2w * np.sum(layer.weights * layer.weights)
                
            # L1 regularization - biases
            if layer.lambda_l1b > 0:
                regularization_loss += layer.lambda_l1b * np.sum(np.abs(layer.biases))
                
            # L2 regularization - biases
            if layer.lambda_l2b > 0:
                regularization_loss += layer.lambda_l2b * np.sum(layer.biases * layer.biases)
            
        return regularization_loss

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

# combined Softmax and Categorical Cross Entropy loss class
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

# binary cross entropy loss class
class BinaryCrossEntropyLoss(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -((y_true * np.log(y_pred_clipped)) + ((1 - y_true) * np.log(1 - y_pred_clipped)))
        sample_losses = np.mean(sample_losses, axis=-1)        
        return sample_losses
    
    # backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues) # number of samples
        n_outputs = len(dvalues[0]) # number of outputs in the sample
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        # calculating gradient
        self.dinputs = -((y_true / dvalues_clipped) - ((1 - y_true) / (1 - dvalues_clipped))) / n_outputs
        self.dinputs = self.dinputs / n_samples
        
# mean squared error loss class
class MeanSquaredErrorLoss(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses
    
    # backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues) # number of samples
        n_outputs = len(dvalues[0]) # number of outputs in every sample
        
        # gradient on values
        self.dinputs = -2 * (y_true - dvalues) / n_outputs
        self.dinputs = self.dinputs / n_samples # normalizing gradient
        
# mean absolute error loss class
class MeanAbsoluteErrorLoss(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    # backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])
        
        # calculating gradient
        self.dinputs = np.sign(y_true - dvalues) / n_outputs
        self.dinputs = self.dinputs / n_samples

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
# per weight adaptive learning rate
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

# adaptive momentum (Adam) optimizer
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
        
    # to be called once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        
        # if layer doesn't have cache and momentum arrays, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # update momentums with current gradients
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums) + ((1 - self.beta_1) * layer.dweights)
        layer.bias_momentums = (self.beta_1 * layer.bias_momentums) + ((1 - self.beta_1) * layer.dbiases)
        
        # get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))
        bias_momentums_corrected = layer.bias_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))
        
        # update cache with squared current gradients
        layer.weight_cache = (self.beta_2 * layer.weight_cache) + ((1 - self.beta_2) * (layer.dweights ** 2))
        layer.bias_cache = (self.beta_2 * layer.bias_cache) + ((1 - self.beta_2) * (layer.dbiases ** 2))
        
        # get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - (self.beta_2 ** (self.iterations + 1))) 
        bias_cache_corrected = layer.bias_cache / (1 - (self.beta_2 ** (self.iterations + 1))) 
        
        # parameter update and normalization w/ sqrt cache
        layer.weights += (-self.current_learning_rate * weight_momentums_corrected) / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += (-self.current_learning_rate * bias_momentums_corrected) / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    # to be called after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
# base accuracy class
class Accuracy:
    # calculating accuracy
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        
        # add accumulated sum of matching values and counts
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy
    
    # calculated accumulated accuracy
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    # reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
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

# accuracy for classification model
class CategoricalAccuracy(Accuracy):
    def __init__(self, binary=False) -> None:
        self.binary = binary # flag for binary classification
        
    # no initialization needed for classification
    # but included bc Model's "train" method calls it
    def init(self, y):
        pass
    
    # comparing predicitons to the ground truth
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# model class
class Model:
    def __init__(self) -> None:
        self.layers = [] # list of network objects
        self.softmax_classifier_output = None
        
    # add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    # set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        
        # if output activation is Softmax and loss is CCE, 
        # create an object of combined activation and loss function
        # that has faster gradient calculation
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            # creating obj of combined activation and loss functions
            self.softmax_classifier_output = SoftmaxActivationCCELoss()
    
    # perform a forward pass
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # "layer" is now last object from the list, return its output
        return layer.output
    
    # perform a backward pass
    def backward(self, output, y):
        
        # when using combined Softmax activation and CCE loss, we have to take a different approach
        if self.softmax_classifier_output is not None:
            # call backward on combo activation/loss to set dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            # since we don't call backward on last layer, we just set its dinputs
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            # call backward method through all objects (except last) in reverse order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
        
        # start by calling backward method on the loss to set dinputs for the last layer to use
        self.loss.backward(output, y)
        
        # call backward method going through all network objects in reverse
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    # training the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # initialize accuracy object
        self.accuracy.init(y)
        
        # setting defaults for step sizes to 1 if batch size is not set
        train_steps = 1 
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        # calculating number of steps when batch size is set
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
                
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        # main training in loop
        for epoch in range(1, epochs + 1):
            
            print()
            print(f"EPOCH: {epoch}")
            
            # reset accumulated loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # iterate over steps
            for step in range(train_steps):
                
                # creating batches from data
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]
                                
                # perform the forward pass
                output = self.forward(batch_X, training=True)
                
                # calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, 
                                                                    include_regularization=True)
                loss = data_loss + regularization_loss
                
                # get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                # perform backward pass
                self.backward(output, batch_y)
                
                # optimize / update params
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                # print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f"step: {step}, " + 
                        f"acc: {accuracy:.3f}, " + 
                        f"loss: {loss:.3f} (data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), " +
                        f"lr: {self.optimizer.current_learning_rate:.5f}")
            
            # print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f"TRAINING: " + 
                  f"acc: {epoch_accuracy:.3f}, " +
                  f"loss: {epoch_loss:.3f} (data loss: {epoch_data_loss:.3f}, reg loss: {epoch_regularization_loss:.3f}), " +
                  f"lr: {self.optimizer.current_learning_rate:.5f}")
            
            # if there is validation data, evaluate the model
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
    
    # evaluates the model using passed in data set
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # default value if batch size parameter is not used
        validation_steps = 1
        
        # calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        # reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        # iterate over steps
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]
        
            # perform forward pass on data, calculate loss, get predictions and accuracy
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        print(f"VALIDATION: acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}")
        
    # retrieves and returns parameters of trainable layers
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters
    
    # update the model with new parameters
    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
            
    # save parameters to a file
    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
            
    # load parameters to a model from a file
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    # save entire model
    def save(self, path):
        # make a deep copy of the current model instance
        model = copy.deepcopy(self)
        
        # reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        # remove data from input layer and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        # for each layer remove input, output, and gradient properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
                
        # save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
    # loads a model from file and returns it as an object
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model