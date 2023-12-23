from activation_functions.activation_function import ActivationFunction
import numpy as np

class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "tanh"

    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1 - np.square(np.tanh(x))