from layers.layer import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self, probability : float, seed : int = 0):
        """
        Dropout layer

        Parameters:
        ---
        probability : float
            Probability of dropping a neuron
        seed : int
            Seed for the random number generator
            No seed is set if seed = 0
        """
        super().__init__()
        self.name = "dropout"

        if not (0 <= probability <= 1):
            print("WARNING: Probability of dropout layer should be in the range [0, 1]. Setting it to 0.5")
            probability = 0.5

        self.probability = probability

        if seed != 0:
            np.random.seed(seed)

    def forward(self, input : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
        self.input = input
        output = np.copy(input)

        if self.probability == 1:
            return np.zeros_like(input)
        
        self.mask = np.random.choice([0, 1], size = input.shape, p = [self.probability, 1 - self.probability])
        output = np.multiply(output, self.mask) / (1 - self.probability)
        
        return output

    def backward(self, output_gradients, learning_rate = None):
        return np.multiply(output_gradients, self.mask) / (1 - self.probability)
