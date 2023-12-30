from layers.activation_layer import ActivationLayer
import numpy as np

class ReLU(ActivationLayer):
    def __init__(self):
        f = lambda x: np.maximum(x, 0)
        # if x > 0: return 1, else if x == 0 return 0.5 else return 0
        df = lambda x: np.where(x > 0, 1, np.where(x == 0, 0.5, 0))

        super().__init__(f, df)
        self.name = "relu"