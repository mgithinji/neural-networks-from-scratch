import numpy as np

# coding a layer of 3 neurons that recieve 4 inputs

# first implementation where each neuron's properties are given a variable
# and calculations are perfomed in rudimentry fashion
def implementation1() -> None:
    inputs = [1, 2, 3, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1.0] # weights for neuron1
    weights2 = [0.5, -0.91, 0.26, -0.5] # weights for neuron2
    weights3 = [-0.26, -0.27, 0.17, 0.87] # weights for neuron3

    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    outputs = [
        # neuron1
        inputs[0] * weights1[0] + 
        inputs[1] * weights1[1] + 
        inputs[2] * weights1[2] + 
        inputs[3] * weights1[3] + bias1,
        
        # neuron2
        inputs[0] * weights2[0] + 
        inputs[1] * weights2[1] + 
        inputs[2] * weights2[2] + 
        inputs[3] * weights2[3] + bias2,

        # neuron3
        inputs[0] * weights3[0] + 
        inputs[1] * weights3[1] + 
        inputs[2] * weights3[2] + 
        inputs[3] * weights3[3] + bias3,
    ]

    print(outputs)

# same calculations, but implemented using programmatic loops to reduce code
# and improve scalability
def implementation2() -> None:
    inputs = [1, 2, 3, 2.5]

    weights = [[0.2, 0.8, -0.5, 1.0], # weights for neuron1
               [0.5, -0.91, 0.26, -0.5], # weights for neuron2
               [-0.26, -0.27, 0.17, 0.87]] # weights for neuron3

    biases = [2, 3, 0.5]
    
    outputs = []
    
    for n_weights, n_bias in zip(weights, biases):
        n_output = 0 # initialize neuron output
        
        # summing all (input x weights) values and adding a bias for each neuron
        for n_input, weight in zip(inputs, n_weights):
            n_output += n_input * weight
        n_output += n_bias

        outputs.append(n_output)
        
    print(outputs)
    
# numpy implementation (best)
def implementation3() -> None:
    inputs = [1, 2, 3, 2.5]

    weights = [[0.2, 0.8, -0.5, 1.0], # weights for neuron1
               [0.5, -0.91, 0.26, -0.5], # weights for neuron2
               [-0.26, -0.27, 0.17, 0.87]] # weights for neuron3

    biases = [2, 3, 0.5]
    
    outputs = np.dot(weights, inputs) + biases
    
    print(outputs)
    
implementation3()