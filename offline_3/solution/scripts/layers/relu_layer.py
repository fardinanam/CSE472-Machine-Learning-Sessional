from layers.activation_layer import ActivationLayer
import numpy as np

class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(self._f_, self._df_)
        self.name = "relu"

    def _f_(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(input, 0)
    
    def _df_(self, input : np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, np.where(input == 0, 0.5, 0))