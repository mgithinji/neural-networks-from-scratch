# coding the softmax derivative

import numpy as np

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1, 1)
print(softmax_output)

# the 1st part of the softmax derivative

print(softmax_output * np.eye(softmax_output.shape[0]))

# simplifying using np.diagflat() method
softmax_output = [0.7, 0.1, 0.2]
print(np.diagflat(softmax_output)) # the 1st part of the softmax derivative

# the 2nd part of the softmax derivative

# showing how we can perfom the 2nd part softmax derivative multiplication with np.dot()
softmax_output = np.array(softmax_output).reshape(-1, 1)
print(np.dot(softmax_output, softmax_output.T))

# the complete softmax derivative in code
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))