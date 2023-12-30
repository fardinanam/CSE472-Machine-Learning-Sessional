import numpy as np
from layers.activation_layer import ActivationLayer

class SoftmaxLayer(ActivationLayer):
    def __init__(self):
        super().__init__(self.__softmax__, None)
        self.name = "softmax"

    def __softmax__(self, x : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
        exp = np.exp(x)
        return exp / np.sum(exp, axis=0)
    
    # def __softmax_derivative__(self, x : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
    #     s = self.__softmax__(x)
    #     print(f"Softmax derivative shape: {np.diagflat(s).shape}")
    #     return np.diagflat(s) - np.dot(s, s.T)
    
    def backward(self, output_gradients: np.ndarray, learning_rate=None) -> np.ndarray:
        input_gradient = self.output * (output_gradients - np.sum(self.output * output_gradients, axis=0))
        return input_gradient