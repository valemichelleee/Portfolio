import numpy as np
from . import Base


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1 , (input_size+1, output_size))
        self._optimizer = None
        self.input = None
        self.error = None
        self._gradient_weight = None

    def initialize(self, weights_initializer, bias_initializer):
        weight = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights = np.concatenate((weight, bias))

    def forward(self, input_tensor):
        input_tensor = np.insert(input_tensor, 0, values=1, axis=1)
        self.input = input_tensor
        return np.dot(self.input, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.error = error_tensor
        self._gradient_weight = np.dot(self.input.T, self.error)
        gradient_input = np.dot(self.error, self.weights[1:, :].T)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weight)
        return gradient_input

    @property
    def gradient_weights(self):
        return self._gradient_weight

