from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    def __init__(self):
        self.name = "activation_function"
        self.input = None
        self.output = None

    @abstractmethod
    def f(self, input : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def df(self, input : np.ndarray) -> np.ndarray:
        pass