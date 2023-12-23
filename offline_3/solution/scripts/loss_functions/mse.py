import numpy as np
from loss_functions.loss_function import LossFunction

class MSE(LossFunction):
    def __init__(self):
        super().__init__()
        self.name = "mse"

    def f(self, y_pred : np.ndarray, y_true : np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_true))

    def df(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size