from nnfs.datasets import spiral_data # non-linear dataset
import numpy as np
import nnfs
import matplotlib.pyplot as plt

# initialize random generators and set types for reproducability
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()