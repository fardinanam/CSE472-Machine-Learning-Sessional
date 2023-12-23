from loss_functions.loss_function import LossFunction
import numpy as np

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.name = "cross_entropy_loss"
        self.epsilon = 1e-8

    def f(self, y_pred : np.ndarray, y_true : np.ndarray) -> float:
        # TODO: Implement the cross entropy loss function (the following implementation is not yet tested)
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(y_true * np.log(y_pred + self.epsilon))
    
    def df(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        # TODO: Implement the derivative of the cross entropy loss function (the following implementation is not yet tested)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + self.epsilon)