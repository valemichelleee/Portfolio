import copy

from . import Base
import numpy as np
from scipy.signal import correlate, correlate2d, convolve


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.input_tensor = None
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None

        if len(convolution_shape) == 2:
            self.conv_flag = 1
            self.weights = np.random.rand(self.num_kernels, convolution_shape[0], convolution_shape[1])
        elif len(convolution_shape) == 3:
            self.conv_flag = 2
            self.weights = np.random.rand(self.num_kernels, convolution_shape[0], convolution_shape[1],
                                          convolution_shape[2])

        self.bias = np.random.rand(self.num_kernels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = None
        batch, channel, height = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
        output_y = int(np.ceil(height / self.stride_shape[0]))

        if self.conv_flag == 1:
            output_tensor = np.zeros((batch, self.num_kernels, output_y))

            for b in range(batch):
                for k in range(self.num_kernels):
                    output_tensor[b][k] += self.bias[k]
                    for c in range(channel):
                        output_tensor[b][k] += correlate(input_tensor[b][c], self.weights[k][c], mode='same')[::self.stride_shape[0]]

        elif self.conv_flag == 2:
            width = input_tensor.shape[3]
            output_x = int(np.ceil(width / self.stride_shape[1]))
            output_tensor = np.zeros((batch, self.num_kernels, output_y, output_x))

            for b in range(batch):
                for k in range(self.num_kernels):
                    output_tensor[b][k] += self.bias[k]
                    for c in range(channel):
                        output_tensor[b][k] += correlate2d(input_tensor[b][c], self.weights[k][c], mode='same')[::self.stride_shape[0], ::self.stride_shape[1]]

        return output_tensor

    def backward(self, error_tensor):
        batch, channel = self.input_tensor.shape[0], self.input_tensor.shape[1]
        gradient_input = np.zeros(self.input_tensor.shape)
        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = np.zeros(self.bias.shape)

        pad_left = (self.convolution_shape[1] - 1) // 2
        pad_right = (self.convolution_shape[1]) // 2

        if self.conv_flag == 1:
            error_tensor_upsampled = np.zeros((batch, self.num_kernels, self.input_tensor.shape[2]))
            self.input_tensor = (np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_left, pad_right))))
            flip_kernel = np.flip(np.transpose(self.weights, (1, 0, 2)), axis=1)

            for b in range(batch):
                for k in range(self.num_kernels):
                    error_tensor_upsampled[b, k, ::self.stride_shape[0]] = error_tensor[b][k]

            for b in range(batch):
                for k in range(self.num_kernels):
                    self._gradient_bias[k] += np.sum(error_tensor_upsampled[b][k])
                    for c in range(channel):
                        self._gradient_weights[k][c] += correlate(self.input_tensor[b][c], error_tensor_upsampled[b][k],
                                                                  mode='valid')

        elif self.conv_flag == 2:
            error_tensor_upsampled = np.zeros((batch, self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))

            pad_top = (self.convolution_shape[2] - 1) // 2
            pad_bottom = self.convolution_shape[2] // 2
            self.input_tensor = np.pad(self.input_tensor,((0, 0), (0, 0), (pad_left, pad_right), (pad_top, pad_bottom)))

            flip_kernel = np.flip(np.transpose(self.weights, (1, 0, 2, 3)), axis=1)

            for b in range(batch):
                for k in range(self.num_kernels):
                    error_tensor_upsampled[b, k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b][k]

            for b in range(batch):
                for k in range(self.num_kernels):
                    self._gradient_bias[k] += np.sum(error_tensor_upsampled[b][k])
                    for c in range(channel):
                        self._gradient_weights[k][c] += correlate2d(self.input_tensor[b][c],
                                                                    error_tensor_upsampled[b][k], mode='valid')

        for i in range(len(error_tensor_upsampled)):
            for j in range(len(flip_kernel)):
                gradient_input[i][j] = convolve(error_tensor_upsampled[i], flip_kernel[j], mode='same')[self.num_kernels // 2]

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * self.convolution_shape[1]
        fan_out = self.num_kernels * self.convolution_shape[1]

        if self.conv_flag == 2:
            fan_in *= self.convolution_shape[2]
            fan_out *= self.convolution_shape[2]

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad_weights):
        self._gradient_weights = grad_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, grad_bias):
        self._gradient_bias = grad_bias

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, _optimizer):
        self._optimizer_weights = copy.deepcopy(_optimizer)
        self._optimizer_bias = copy.deepcopy(_optimizer)
