import numpy as np
from . import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input = None

    def forward(self, input_tensor):
        input_tensor -= np.max(input_tensor)
        sums = np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        self.input = np.exp(input_tensor)/sums
        return self.input

    def backward(self, error_tensor):
        weighted_sum = np.sum(np.multiply(error_tensor, self.input), axis=1, keepdims=True)
        adjusted_error = error_tensor - weighted_sum
        return np.multiply(self.input, adjusted_error)

