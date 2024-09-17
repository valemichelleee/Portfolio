import numpy as np


class TanH:
    def __init__(self):
        self.activation = None
        self.trainable = False

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activation**2)
