import numpy as np
from learner import Learner
import math

class AdaBoost(Learner):
    def __init__(self, learner : Learner, error_threshold : float = 0.5) -> None:
        """
        Parameters:
        -----------
        learner : Learner
            The learner to use for the boosting
        error_threshold : float
            The error threshold to use for the boosting
        """
        self.learner = learner
        self.error_threshold = error_threshold
        self.h = [] # list of weak learners
        self.z = [] # list of weak learner weights
    
    def fit(self, X : np.ndarray, y : np.ndarray, epochs:int=1000) -> Learner:
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            The training data.
        y : np.ndarray
            The labels.
        epochs : int
            The number of epochs to train for
        
        Returns:
        --------
        self : AdaBoost
            The fitted model
        """
        n = X.shape[0] # number of examples
        w = np.ones((n, 1)) / n # vector of N example weights

        epsilon = 1e-7 # to avoid division by zero

        np.random.seed(51)
        for _ in range(epochs):
            # resample the data
            indices = np.random.choice(n, n, p=w.flatten())
            X_resampled = X[indices]
            y_resampled = y[indices]

            learner = self.learner()
            # learner.fit(X_resampled, y_resampled, epochs=1000)
            learner.fit(X_resampled, y_resampled)
            self.h.append(learner)

            error = 0

            for i in range(n):
                X_i = np.reshape(X[i], (1, -1))
                if learner.predict(X_i) != y[i]:
                    error += w[i].flatten()[0]

            if error > self.error_threshold:
                self.z.append(math.log((1 - self.error_threshold) / self.error_threshold))
                continue
            
            # error = min(error, 1 - epsilon)

            for i in range(n):
                X_i = np.reshape(X[i], (1, -1))
                if learner.predict(X_i) == y[i]:
                    w[i] *= error / (1 - error)
            
            # normalize the weights
            w /= w.sum()

            self.z.append(math.log((1 - error) / (error + epsilon)))
        
        return self

    def predict(self, x : np.ndarray) -> np.ndarray:
        """
        Predict the label for a single example.
        
        Parameters:
        -----------
        x : np.ndarray
            The example to predict
        
        Returns:
        --------
        y : np.ndarray
            The predicted label having values in {-1, 1}
        """
        z = np.array(self.z)
        z = z / z.sum()
        y_pred = np.zeros(x.shape[0])

        for i in range(len(self.h)):
            y_pred += z[i] * self.h[i].predict(x).flatten()

        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred            
