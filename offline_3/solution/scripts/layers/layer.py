import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.name = "layer"
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
        pass

    @abstractmethod
    def backward(self, output_gradients : np.ndarray[any, np.dtype[float]], learning_rate : float) -> np.ndarray[any, np.dtype[float]]:
        pass