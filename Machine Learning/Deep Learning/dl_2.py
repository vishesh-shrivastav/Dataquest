import matplotlib.pyplot as plt
%matplotlib inline

# ReLu function
import numpy as np
x = np.linspace(-2, 2, 20)

def relu(values):
    return np.maximum(x, 0)

relu_y = relu(x)

print(x)
print(relu_y)

plt.plot(x, relu_y)

# Plotting the tan function
x = np.linspace(-2*np.pi, 2*np.pi, 100)
tan_y = np.tan(x)
plt.plot(x, tan_y)

# Plotting the hyperbolic tangent function
x = np.linspace(-40, 40, 100)
tanh_y = np.tanh(x)
plt.plot(x, tanh_y)