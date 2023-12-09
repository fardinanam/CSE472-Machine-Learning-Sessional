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
    feature = X[:, feature_index]
    n = y.size
    unique_values, counts = np.unique(feature, return_counts=True)
    
    if unique_values.size > 10:
        median = np.median(unique_values)
        
        # split the feature into two values
        values_less = feature[feature <= median].flatten()
        values_greater = feature[feature > median].flatten()
        
        # map the values of feature to 0 and 1
        feature = np.where(feature <= median, 0, 1)
        counts = np.array([values_less.size, values_greater.size])
        unique_values = np.array([0, 1])
    
    remainder = 0
    for i in range(unique_values.size):
        indices = np.where(feature == unique_values[i])[0]
        n_v = counts[i]
        y_v = y[indices]
        entropy_v = entropy(y_v)
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
