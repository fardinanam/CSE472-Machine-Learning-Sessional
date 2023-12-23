from layers.activation_layer import ActivationLayer
import numpy as np

class Tanh(ActivationLayer):
    def __init__(self):
        f = lambda x: np.tanh(x)
        df = lambda x: 1 - np.square(np.tanh(x))

        super().__init__(f, df)
        self.name = "tanh"