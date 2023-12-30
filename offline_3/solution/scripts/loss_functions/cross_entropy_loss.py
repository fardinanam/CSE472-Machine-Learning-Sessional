from loss_functions.loss_function import LossFunction
import numpy as np

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.name = "cross_entropy_loss"
        self.epsilon = 1e-8
    
    def softmax(self, x : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
        # check if x contains NaN or Inf
        # if np.isnan(x).any():
        #     # raise ValueError("x contains NaN")
        #     print(x)

        # if np.isinf(x).any():
        #     # raise ValueError("x contains Inf")
        #     print(x)

        x = x - np.max(x, axis=0)
        exp = np.exp(x)
        return exp / (np.sum(exp, axis=0) + self.epsilon)

    def f(self, y_pred : np.ndarray[any, np.dtype[float]], y_true : np.ndarray[any, np.dtype[float]]) -> float:
        """
        Cross entropy loss function

        Parameters:
        ---
        y_pred (np.ndarray): predicted values (without softmax)
        y_true (np.ndarray): true values

        Returns:
        ---
        float: cross entropy loss after applying softmax to y_pred
        """
        s = self.softmax(y_pred)
        return -np.sum(y_true * np.log(s + self.epsilon))
    
    def df(self, y_pred : np.ndarray[any, np.dtype[float]], y_true : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
        """
        Derivative of cross entropy loss with respect to y_pred

        Parameters:
        ---
        y_pred (np.ndarray): predicted values (without softmax)
        y_true (np.ndarray): true values

        Returns:
        ---
        np.ndarray: derivative of cross entropy loss with respect to y_pred
        """
        s = self.softmax(y_pred)
        return s - y_true