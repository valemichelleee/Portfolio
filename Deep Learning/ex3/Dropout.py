import numpy as np
from . import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.trainable = False
        self.testing_phase = False
        self.flag = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.flag = np.random.binomial(1, self.probability, size=input_tensor.shape)
            return input_tensor * self.flag / self.probability

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        else:
            return error_tensor * self.flag / self.probability
