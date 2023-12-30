from layers.layer import Layer
from activation_functions.activation_function import ActivationFunction
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, f : callable, df : callable):
        super().__init__()
        self.name = "activation"
        self.f = f
        self.df = df

    def forward(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        self.output = self.f(input)

        return self.output

    def backward(self, output_gradients : np.ndarray, learning_rate = None) -> np.ndarray:
        # print(f"Output gradients shape: {output_gradients.shape}")
        return np.multiply(output_gradients, self.df(self.input))