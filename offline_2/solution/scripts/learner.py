from abc import ABC, abstractmethod
import numpy as np

def sigmoid(z : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

def losses(h : np.ndarray, y : np.ndarray) -> np.ndarray:
    return -y * np.log(h) - (1 - y) * np.log(1 - h)

class Learner(ABC):
    @abstractmethod
    def __init__(self, learning_rate:float=0.5, error_threshold:float=0.5) -> None:
        pass

    @abstractmethod
    def fit(self, X_train, y_train, init_W, epochs):
        """
        Fit the model to the data.

        Parameters:
        -----------
        X : np.ndarray
            The training data.
        y : np.ndarray
            The labels.
        init_W : np.ndarray
            The initial weights to use for the model
        epochs : int
            The number of epochs to train for
        """
        pass

    @abstractmethod
    def predict(self, x):
        pass