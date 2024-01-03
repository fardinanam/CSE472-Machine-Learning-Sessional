from activation_functions.sigmoid import Sigmoid as Sig
from layers.activation_layer import ActivationLayer
import numpy as np

class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(self._f_, self._df_)
        self.name = "sigmoid"

    def _f_(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        return 1 / (1 + np.exp(-input))
    
    def _df_(self, input : np.ndarray) -> np.ndarray:
        s = self._f_(input)
        return s * (1 - s)