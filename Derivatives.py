import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2

def tangent_line(x, m, b):
    return (m * x) + b

x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c']

for i in range(5):

    delta = 0.0001
    
    x1 = i
    x2 = x1 + delta

    y1 = f(x1)
    y2 = f(x2)
    
    print(f"({x1}, {y1}) ({x2}, {y2})")

    m = (y2 - y1) / (x2 - x1) # approximate derivative
    b = y2 - (m * x2)

    plot_range = [x1 - 0.9, x1, x1 + 0.9]
    
    plt.scatter(x1, y1, c=colors[i])

    plt.plot([j for j in plot_range], 
             [tangent_line(x=j, m=m, b=b) for j in plot_range], 
             c=colors[i])

    print(f"approximate derivative for f(x) = 2x^2 at x = {x1}: {m}")

plt.show()