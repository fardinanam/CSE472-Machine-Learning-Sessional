import numpy as np
from optimizers.optimizer import Optimizer
from layers.dense_layer import DenseLayer

class AdamOptimizer(Optimizer):
    def __init__(self, layers : list[DenseLayer], learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self):

        if self.m is None:
            self.m = {}
            self.v = {}
            
            for i, layer in enumerate(self.layers):
                self.m[f"{layer.name}_{i+1}_weights"] = np.zeros(layer.weights.shape)
                self.m[f"{layer.name}_{i+1}_biases"] = np.zeros(layer.biases.shape)
                self.v[f"{layer.name}_{i+1}_weights"] = np.zeros(layer.weights.shape)
                self.v[f"{layer.name}_{i+1}_biases"] = np.zeros(layer.biases.shape)

        self.t += 1

        for i, layer in enumerate(self.layers):
            weights_name = f"{layer.name}_{i+1}_weights"
            biases_name = f"{layer.name}_{i+1}_biases"

            beta1_t = self.beta1 ** self.t
            beta2_t = self.beta2 ** self.t
            # learning_rate = self.learning_rate * np.sqrt(1 - beta2_t) / (1 - beta1_t)
            learning_rate = self.learning_rate

            self.m[weights_name] = self.beta1 * self.m[weights_name] + (1 - self.beta1) * layer.weights_gradients
            self.v[weights_name] = self.beta2 * self.v[weights_name] + (1 - self.beta2) * np.square(layer.weights_gradients)

            m_hat = self.m[weights_name] / (1 - beta1_t)
            v_hat = self.v[weights_name] / (1 - beta2_t)

            layer.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.m[biases_name] = self.beta1 * self.m[biases_name] + (1 - self.beta1) * layer.biases_gradients
            self.v[biases_name] = self.beta2 * self.v[biases_name] + (1 - self.beta2) * np.square(layer.biases_gradients)

            m_hat = self.m[biases_name] / (1 - beta1_t)
            v_hat = self.v[biases_name] / (1 - beta2_t)

            layer.biases -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)