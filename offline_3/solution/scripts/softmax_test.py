from layers.softmax_layer import SoftmaxLayer
import numpy as np

# test
    
x = np.array([1, 2])

s = SoftmaxLayer()
print(s.forward(x))

print(s.backward(x, 0.1))