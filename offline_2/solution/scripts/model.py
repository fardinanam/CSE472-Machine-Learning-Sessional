from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split(self, test_size : float = 0.2, random_state : int = 51) -> list:
        pass

    @abstractmethod
    def select_k_features(self, k : int):
        pass

    def train(self):
        pass

    def predict(self, X_test : np.ndarray) -> np.ndarray:
        pass
