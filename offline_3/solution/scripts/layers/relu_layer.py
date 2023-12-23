from layers.activation_layer import ActivationLayer
import numpy as np

class ReLU(ActivationLayer):
    def __init__(self):
        f = lambda x: np.maximum(x, 0)
        df = lambda x: np.where(x > 0, 1, 0)

        super().__init__(f, df)
        self.name = "relu"