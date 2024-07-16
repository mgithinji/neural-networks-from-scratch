import numpy as np

# Coding a single neuron
# 4 inputs to the neuron, 4 weights assoc w/ each input, and a single bias

# rudimentary implementation
def implementation1() -> None:
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2

    # summing all (input x weights) values and adding a bias
    output = (inputs[0] * weights[0] + 
            inputs[1] * weights[1] + 
            inputs[2] * weights[2] + 
            inputs[3] * weights[3] + bias)

    print(output)

# numpy implmentation
def implementation2() -> None:
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2
    
    output = np.dot(weights, inputs) + bias
    
    print(output)
    
implementation2()