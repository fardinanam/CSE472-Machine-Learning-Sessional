from activation_functions.activation_function import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "sigmoid"
    
    def f(self, input : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
    
    def df(self, input : np.ndarray) -> np.ndarray:
        s = self.forward(input)
        return s * (1 - s)