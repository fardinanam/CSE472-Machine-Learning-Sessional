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
        w = np.ones(n) / n # vector of N example weights

        epsilon = 1e-6 # to avoid division by zero

        for _ in range(epochs):
            # resample the data
            indices = np.random.choice(n, n, p=w)
            X_resampled = X[indices]
            y_resampled = y[indices]

            learner = self.learner()
            learner.fit(X_resampled, y_resampled, epochs=1000)
            self.h.append(learner)

            error = 0

            for i in range(n):
                if learner.predict(X[i]) != y[i]:
                    error += w[i]

            if error > self.error_threshold:
                continue
            
            error = min(error, 1 - epsilon)

            for i in range(n):
                if learner.predict(X[i]) != y[i]:
                    w[i] *= error / (1 - error)
            
            # normalize the weights
            w /= w.sum()

            self.z.append(math.log((1 - error) / (error + epsilon)))
        
        return self

    def predict(self, x):
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
        y_preds = np.array([h.predict(x) for h in self.h])
        # self.z = np.array(self.z)

        # print(f"z shape: {np.array(self.z).shape}")
        # print(f"z: {self.z}")
        # print(f"y_preds shape: {y_preds.shape}")
        # print(f"y_preds: {y_preds}")

        return np.sign(np.dot(self.z, y_preds))


            



