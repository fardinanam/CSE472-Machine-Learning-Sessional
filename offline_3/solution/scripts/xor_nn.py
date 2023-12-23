import numpy as np
from layers.dense_layer import DenseLayer
from layers.relu_layer import ReLU
from layers.tanh_layer import Tanh
from loss_functions.cross_entropy_loss import CrossEntropyLoss
from loss_functions.mse import MSE

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    DenseLayer(2, 3),
    Tanh(),
    DenseLayer(3, 1),
    Tanh()
]

loss = MSE()

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += loss.f(output, y)
        output_gradient = loss.df(output, y)

        for layer in reversed(network):
            output_gradient = layer.backward(output_gradient, learning_rate)

    error /= len(X)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} loss: {error}")

for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    
    output = np.round(output)
    print(f"Input: {x} Output: {output} Expected: {y}")
