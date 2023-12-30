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

    def forward(self, input : np.ndarray[any, np.dtype[float]]) -> np.ndarray[any, np.dtype[float]]:
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
        # print(f"Input shape: {self.input.shape}")
        # print(f"Weights shape: {self.weights.shape}")
        self.output = np.dot(self.weights, input) + self.biases

        return self.output

    def backward(self, output_gradients: np.ndarray[any, np.dtype[float]], learning_rate: float) -> np.ndarray[any, np.dtype[float]]:
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
        self.weights -= np.dot(output_gradients, self.input.T) * learning_rate
        self.biases -= learning_rate * np.sum(output_gradients, axis=1, keepdims=True)

        return np.dot(self.weights.T, output_gradients)
    

DenseLayer(3, 2)