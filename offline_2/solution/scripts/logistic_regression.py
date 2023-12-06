import numpy as np
from learner import Learner, sigmoid, losses

class LogisticRegression(Learner):
    def __init__(self, alpha:float=0.5, error_threshold:float=0.5) -> None:
        self.W = None
        self.alpha = alpha
        self.error_threshold = error_threshold

    def fit(self, X : np.ndarray, y : np.ndarray, init_W : np.ndarray = [], epochs:int=1000) -> np.ndarray:
        if init_W != []:
            self.W = init_W
        else:
            self.W = np.zeros(X.shape[1])

        for _ in range(epochs):
            z = np.dot(X, self.W)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.W -= self.alpha * gradient

            if self.error_threshold != 0. and losses(h, y).mean() < self.error_threshold:
                break

        return self
    
    def predict(self, x : np.ndarray) -> np.ndarray:
        y_pred = sigmoid(np.dot(x, self.W))
        # convert probabilities to binary output values
        return np.where(y_pred >= 0.5, 1, 0)