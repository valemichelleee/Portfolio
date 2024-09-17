import numpy as np
from . import Base


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input_tensor):
        self.input = input_tensor
        return np.maximum(0, self.input)

    def backward(self, error_tensor):
        gradient_tensor = self.input > 0
        return error_tensor * gradient_tensor

