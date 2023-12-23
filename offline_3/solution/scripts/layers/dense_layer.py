from layers.layer import Layer
import numpy as np

class DenseLayer(Layer):
    def __init__(self, input_size : int, output_size : int, seed : int = 0):
        super().__init__()
        self.name = "dense"
        self.input_size = input_size
        self.output_size = output_size

        if seed != 0:
            np.random.seed(seed)

        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input : np.ndarray) -> np.ndarray:
        """
        Forward pass of the dense layer

        Parameters:
        ---
        input : np.ndarray of shape (input_size, batch_size)
            Input to the layer
        
        Returns:
        ---
        np.ndarray of shape (output_size, batch_size)
            Output of the layer
        """
        self.input = input
        self.output = np.dot(self.weights, input) + self.biases
        return self.output

    def backward(self, output_gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass of the dense layer

        Parameters:
        ---
        output_gradients : np.ndarray of shape (output_size, batch_size)
            Gradients flowing out from the layer
        learning_rate : float
            Learning rate of the optimizer

        Returns:
        ---
        np.ndarray of shape (input_size, batch_size)
            Gradients flowing into the layer
        """
        self.weights -= learning_rate * np.dot(output_gradients, self.input.T)
        self.biases -= learning_rate * output_gradients

        return np.dot(self.weights.T, output_gradients)
    

DenseLayer(3, 2)