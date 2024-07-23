# coding the backpropagation method for a single neuron
# NOTE: this is a rudimentary example to demonstrate the
# how partial derivatives are backpropagated

# single neuron with three inputs (and 3 weights) and ReLU activation

# single neuron receiving three inputs (forward pass)
x = [1.0, -2.0, 3.0] # inputs
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)

# adding weighted inputs and a bias (neuron's output)
z = xw0 + xw1 + xw2 + b
print(z)

# ReLU activation on the neuron's output
y = max(z, 0)
print(y)

# backward pass

# derivative value from the next layer (arbitrarily chosen)
dval = 1.0

# derivative of ReLU and the chain rule
drelu_dz = dval * (1.0 if z > 0 else 0.0)
print(drelu_dz)

dsum_dxw0 = 1 # partial derivative of a sum(x,y) operation is always 1
dsum_dxw1 = 1 
dsum_dxw2 = 1 
dsum_db = 1

drelu_dxw0 = drelu_dz * dsum_dxw0 # partial derivative of ReLU wrt the [0] mul(x,w) pair
drelu_dxw1 = drelu_dz * dsum_dxw1 # partial derivative of ReLU wrt the [1] mul(x,w) pair
drelu_dxw2 = drelu_dz * dsum_dxw2 # partial derivative of ReLU wrt the [2] mul(x,w) pair
drelu_db = drelu_dz * dsum_db # partial derivative of ReLU wrt the bias
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# if f(x) = Ax, d'(x) = A -- partial derivative wrt x results in the multiplicative constant
# taking partials wrt x[...] and w[...]
dmul_dx0 = w[0] 
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0 # partial derivative of ReLU wrt x[...]
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0 # partial derivative of ReLU wrt w[...]
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradients on weights
db = drelu_db # gradient on bias

print(w, b)

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b)

# another forward pass

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
z = xw0 + xw1 + xw2 + b
y = max(z, 0)
print(y)