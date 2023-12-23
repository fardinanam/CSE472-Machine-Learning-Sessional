from activation_functions.activation_function import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "relu"

    def f(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(input, 0)
    
    def df(self, input : np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, 0)