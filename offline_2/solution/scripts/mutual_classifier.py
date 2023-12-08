import numpy as np

def entropy(y : np.ndarray) -> float:
    # find the number of unique classes
    classes = np.unique(y)
    # find the number of examples
    n = y.size
    # find the entropy
    entropy = 0
    for c in classes:
        p = np.sum(y == c) / n
        entropy -= p * np.log2(p)
    return entropy

def remainder(X : np.ndarray, y : np.ndarray, feature_index : int) -> float:
    # find the number of unique classes
    classes = np.unique(y)
    # find the number of examples
    n = y.size
    # find the number of features
    n_features = X.shape[1]
    # find the number of unique values of the feature
    values = np.unique(X[:, feature_index])
    # find the remainder
    remainder = 0
    for v in values:
        # find the indices of the examples with the value v
        indices = np.where(X[:, feature_index] == v)[0]
        # find the number of examples with the value v
        n_v = indices.size
        # find the entropy of the examples with the value v
        y_v = y[indices]
        entropy_v = entropy(y_v)
        # find the remainder
        remainder += (n_v / n) * entropy_v
    return remainder

def gain(X : np.ndarray, y : np.ndarray, feature_index : int) -> float:
    # find the entropy of y
    y_entropy = entropy(y)
    # find the remainder of the feature
    feature_remainder = remainder(X, y, feature_index)
    # find the gain
    gain = y_entropy - feature_remainder
    return gain

def info_gain_score(X : np.ndarray, y : np.ndarray) -> np.ndarray:
    # find the number of features
    n_features = X.shape[1]

    # find the gain of each feature
    scores_ = np.zeros(n_features)

    for i in range(n_features):
        scores_[i] = gain(X, y, i)

    return scores_  

    

class SelectKBest():
    def __init__(self, score_func = info_gain_score, k : int=10):
        self.score_func = score_func
        self.k = k
        self.scores_ = []

    def fit(self, X : np.ndarray, y : np.ndarray):
        self.scores_ = self.score_func(X, y)
        return self

    def transform(self, X):
        indices = np.argsort(self.scores_)[::-1][:self.k]
        return X[:, indices]
