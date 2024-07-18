# coding the loss functions

import numpy as np

# base Loss class
class Loss:
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

# testing my loss function
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

target_classes = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])

loss_fn = CategoricalCrossEntropyLoss()
loss = loss_fn.calculate(output=softmax_outputs, y=target_classes)
print(loss)