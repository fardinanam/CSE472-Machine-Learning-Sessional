from abc import ABC, abstractmethod
from models.model import Model
from layers.layer import Layer

class Optimizer(ABC):
    def __init__(self, layers : list[Layer] = None):
        self.layers = layers

    @abstractmethod
    def step(self):
        pass