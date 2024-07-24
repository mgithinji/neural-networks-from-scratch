# testing the backward functions implemented in our Softmax and Loss classes
from NeuralNetwork import DenseLayer, ReLU, Softmax, CategoricalCrossEntropyLoss, SoftmaxActivationCCELoss, calculate_accuracy
import numpy as np
from timeit import timeit

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

def f1():
    softmax_loss = SoftmaxActivationCCELoss()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs
    return dvalues1

def f2():
    activation = Softmax()
    activation.output = softmax_outputs
    loss = CategoricalCrossEntropyLoss()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs
    return dvalues2

print("gradients: combined loss and activation")
print(f1())
print("gradients: separate loss and activation")
print(f2())

# comparing the speeds of the two approaches (combined vs separate loss and activation functions)
t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print("timings: combined loss and activation")
print(t1)
print("timings: separate loss and activation")
print(t2)
print("timing (difference): separate / combined")
print(t2 / t1)

