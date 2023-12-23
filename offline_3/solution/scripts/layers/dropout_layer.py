from layers.layer import Layer

class DropoutLayer(Layer):
    def __init__(self, probability):
        NotImplementedError

    def forward(self, input):
        NotImplementedError

    def backward(self, output_gradients, learning_rate = None):
        NotImplementedError