# coding the backpropagation of ReLU activation function
# derivative of ReLU -- 1: when input > 0; 
#                       0: otherwise

import numpy as np

# example layer output
z = np.array([[1, 2, -3, 4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# ReLU activation function's derivative
drelu = np.zeros_like(z)
drelu[z > 0] = 1
# applying chain rule
drelu = drelu * dvalues
print(drelu)

# simplyfing the above operations, zeroing all inputs <= 0
drelu = dvalues.copy()
drelu[z <= 0] = 0
print(drelu)