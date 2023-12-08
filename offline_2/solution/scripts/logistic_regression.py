import numpy as np
from learner import Learner, sigmoid, losses

class LogisticRegression(Learner):
    def __init__(self, alpha:float=0.01, error_threshold:float=0.5) -> None:
        self.W = None
        self.alpha = alpha
        self.error_threshold = error_threshold

    def fit(self, X : np.ndarray, y : np.ndarray, epochs:int=1000) -> np.ndarray:
        # check if x and y are 1D arrays
        if len(X.shape) == 1:
            X = np.reshape(X, (X.size, 1))
        if len(y.shape) == 1:
            y = np.reshape(y, (y.size, 1))

        # print("X shape: ", X.shape, "y shape: ", y.shape)
        self.W = np.zeros((X.shape[1], 1))
        # print("W shape: ", self.W.shape)

        for _ in range(epochs):
            z = np.dot(X, self.W)
            # print("z shape: ", z.shape)
            h = sigmoid(z)

            # print("h shape: ", h.shape, "y shape: ", y.shape)

            diff = h - y
            gradient = np.dot(X.T, diff) / y.size

            # print("gradient shape: ", gradient.shape)
            # print("W shape: ", self.W.shape)
            self.W = np.reshape(self.W, (self.W.size, 1))
            self.W = self.W - self.alpha * gradient

            if self.error_threshold != 0. and losses(h, y).mean() < self.error_threshold:
                break

        return self
    
    def predict(self, x : np.ndarray) -> np.ndarray:
        y_pred = sigmoid(np.dot(x, self.W))
        # convert probabilities to binary output values
        return np.where(y_pred >= 0.5, 1, 0)