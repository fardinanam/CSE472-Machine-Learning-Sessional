from layers.activation_layer import ActivationLayer
import numpy as np

class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(self._f_, self._df_)
        self.name = "tanh"

    def _f_(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        return np.tanh(input)
    
    def _df_(self, input : np.ndarray) -> np.ndarray:
        return 1 - np.square(np.tanh(input))