import numpy as np
from layers.activation_layer import ActivationLayer
from layers.layer import Layer
from layers.dropout_layer import DropoutLayer
from loss_functions.loss_function import LossFunction
from tqdm import tqdm

class Model:
    def __init__(self, layers : list[Layer], loss : LossFunction):
        self.layers = layers
        self.loss = loss
        self.name = "model"
    
    def forward(self, input : np.ndarray, is_training = True) -> np.ndarray:
        output = input
        for i, layer in enumerate(self.layers):
            if not is_training and isinstance(layer, DropoutLayer):
                continue
            
            output = layer.forward(output)
            
            # check if any value is nan or inf
            if np.isnan(output).any():
                raise ValueError(f"output contains NaN in forward of {i}: {layer.name} layer")
            if np.isinf(output).any():
                raise ValueError(f"output contains Inf in forward of {i}: {layer.name} layer")
        
        return output
    
    def backward(self, output_gradients : np.ndarray, learning_rate = None) -> np.ndarray:
        gradients = output_gradients
        for layer in reversed(self.layers):
            # print(f"Backwarding through {layer.name} layer")
            gradients = layer.backward(gradients, learning_rate)
            # print(f"Gradients shape: {gradients.shape}")

            # check if any value is nan or inf
            if np.isnan(gradients).any():
                raise ValueError(f"gradients contains NaN in backward of {layer.name} layer")
            if np.isinf(gradients).any():
                raise ValueError(f"gradients contains Inf in backward of {layer.name} layer")
        return gradients
    
    def train(self, input : np.ndarray, output : np.ndarray, learning_rate : float, epochs : int = 10000) -> float:
        for epoch in tqdm(range(epochs)):
            error = 0
            for x, y in zip(input, output):
                output = self.forward(x)
                error += self.loss.f(output, y)
                output_gradient = self.loss.df(output, y)
                self.backward(output_gradient, learning_rate)
            
            error /= len(input)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} loss: {error}")
        
        return error
    
    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.forward(input, False)

