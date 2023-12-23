import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.name = "layer"
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradients : np.ndarray, learning_rate : float) -> np.ndarray:
        pass