from layers.layer import Layer
import numpy as np

class DenseLayer(Layer):
    def __init__(self, input_size : int, output_size : int, seed : int = 0):
        assert input_size > 0, "Input size must be greater than 0"
        assert output_size > 0, "Output size must be greater than 0"

        super().__init__()
        self.name = "dense"
        self.input_size = input_size
        self.output_size = output_size

        if seed != 0:
            np.random.seed(seed)

        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, size=(output_size, input_size))
        self.biases = np.zeros((output_size, 1))

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
        self.input_gradients = np.dot(self.weights.T, output_gradients)
        self.weights_gradients = np.dot(output_gradients, self.input.T)
        self.biases_gradients = np.sum(output_gradients, axis=1, keepdims=True)

        return self.input_gradients
    