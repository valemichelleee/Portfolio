import numpy as np
from . import Base
from .Helpers import compute_bn_gradients


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.bias = None
        self.weights = None
        self.epsilon = np.finfo(np.float64).eps
        self.mean = None
        self.var = None
        self.moving_mean = None
        self.moving_var = None
        self.momentum = 0.8
        self.input_tensor, self.input_shape = None, None
        self.norm_input = None
        self._optimizer = None
        self._gradient_bias, self._gradient_weights = None, None
        self.initialize()

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape

        reshaped_input = self.reformat(input_tensor)

        if not self.testing_phase:
            self.mean = np.mean(reshaped_input, axis=0, keepdims=True)
            self.var = np.var(reshaped_input, axis=0, keepdims=True)
            if self.moving_mean is None and self.moving_var is None:
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
                self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
            self.norm_input = (reshaped_input - self.mean) / np.sqrt(self.var + self.epsilon)
            output = self.weights * self.norm_input + self.bias
        else:
            self.norm_input = (reshaped_input - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
            output = self.weights * self.norm_input + self.bias

        return self.reformat(output)

    def backward(self, error_tensor):
        reshaped_error_tensor = self.reformat(error_tensor)
        reshaped_input_tensor = self.reformat(self.input_tensor)

        grad_input = compute_bn_gradients(reshaped_error_tensor, reshaped_input_tensor, self.weights, self.mean, self.var)
        reshaped_grad_input = self.reformat(grad_input)

        self._gradient_weights = np.sum(reshaped_error_tensor * self.norm_input, axis=0)
        self._gradient_bias = np.sum(reshaped_error_tensor, axis=0)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

        return reshaped_grad_input

    def reformat(self, tensor):
        if len(tensor.shape) != 2:  # Reformat to 2D for normalization
            if len(tensor.shape) == 4:
                tensor = tensor.transpose(0, 2, 3, 1).reshape(-1, self.channels)
            elif len(tensor.shape) == 3:
                tensor = tensor.reshape(-1, self.channels)
            return tensor
        else:  # Reformat back to original shape
            if len(self.input_shape) == 4:  # Image-like tensor
                tensor = tensor.reshape(self.input_shape[0], self.input_shape[2], self.input_shape[3], self.channels).transpose(0, 3, 1, 2)
            elif len(self.input_shape) == 3:  # Sequence-like tensor
                tensor = tensor.reshape(self.input_shape[0], self.input_shape[1], self.channels)
            return tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
