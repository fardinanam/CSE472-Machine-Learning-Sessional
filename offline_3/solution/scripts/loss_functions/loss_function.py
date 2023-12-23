from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    def __init__(self):
        self.name = "loss_function"
        self.input = None
        self.output = None

    @abstractmethod
    def f(self, input : np.ndarray, target : np.ndarray) -> float:
        pass

    @abstractmethod
    def df(self, input : np.ndarray, target : np.ndarray) -> np.ndarray:
        pass