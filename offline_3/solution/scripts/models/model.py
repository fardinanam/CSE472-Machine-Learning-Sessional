import numpy as np
from layers.layer import Layer
from layers.dense_layer import DenseLayer
from layers.dropout_layer import DropoutLayer
from loss_functions.loss_function import LossFunction
import pickle

class Model:
    def __init__(self, layers : list[Layer], loss : LossFunction, name : str = "model"):
        self.layers = layers
        self.loss = loss
        self.name = name
    
    def denselayers(self) -> list[DenseLayer]:
        return [layer for layer in self.layers if isinstance(layer, DenseLayer)]
    
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
    
    def backward(self, output_gradients : np.ndarray) -> np.ndarray:
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients, learning_rate = None)

            # check if any value is nan or inf
            if np.isnan(gradients).any():
                raise ValueError(f"gradients contains NaN in backward of {layer.name} layer")
            if np.isinf(gradients).any():
                raise ValueError(f"gradients contains Inf in backward of {layer.name} layer")
        return gradients
    
    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.forward(input, False)
    
    def save(self, filename : str):
        for layer in self.layers:
            layer.input = None
            layer.output = None

            if isinstance(layer, DenseLayer):
                layer.input_gradients = None
                layer.weights_gradients = None
                layer.biases_gradients = None

            if isinstance(layer, DropoutLayer):
                layer.mask = None

        with open(filename, "wb") as f:
            pickle.dump(self, f)

