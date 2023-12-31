import numpy as np
from layers.layer import Layer
from layers.dense_layer import DenseLayer
from layers.dropout_layer import DropoutLayer
from loss_functions.loss_function import LossFunction

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

            # if isinstance(layer, DenseLayer):
            #     layer.weights = self.optimizer.update(layer.weights_gradients, layer.weights)

            #     print(f"Layer: {layer.name} biases gradients: {layer.biases_gradients.shape}")
            #     print(f"Layer: {layer.name} biases: {layer.biases.shape}")
            #     layer.biases = self.optimizer.update(layer.biases_gradients, layer.biases)

            # check if any value is nan or inf
            if np.isnan(gradients).any():
                raise ValueError(f"gradients contains NaN in backward of {layer.name} layer")
            if np.isinf(gradients).any():
                raise ValueError(f"gradients contains Inf in backward of {layer.name} layer")
        return gradients
    
    # def train(self, input : np.ndarray, output : np.ndarray, learning_rate : float, epochs : int = 10000) -> float:
    #     for epoch in tqdm(range(epochs)):
    #         error = 0
    #         for x, y in zip(input, output):
    #             output = self.forward(x)
    #             error += self.loss.f(output, y)
    #             output_gradient = self.loss.df(output, y)
    #             self.backward(output_gradient, learning_rate)
            
    #         error /= len(input)
    #         if epoch % 1000 == 0:
    #             print(f"Epoch {epoch} loss: {error}")
        
    #     return error
    
    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.forward(input, False)

