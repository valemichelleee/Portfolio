import numpy as np


class Sigmoid:
    def __init__(self):
        self.activation = None
        self.trainable = False

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * self.activation * (1 - self.activation)
