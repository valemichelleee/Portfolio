import copy

import numpy as np
from .TanH import TanH
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected
from .Base import BaseLayer


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.trainable = True
        self._memorize = False
        self.hidden_state = np.zeros((1, hidden_size))

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.fc_hidden = FullyConnected(input_size+hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.inputs = []
        self.hidden_states = []
        self.input_tensor = None
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_weights_output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch = input_tensor.shape[0]

        if not self._memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))

        self.inputs = []
        self.hidden_states = []
        self.hidden_states.append(self.hidden_state)
        output = []

        for t in range(batch):
            input_t = input_tensor[t].reshape(1, -1)
            self.inputs.append(input_t)

            combined_input = np.concatenate((input_t, self.hidden_state.reshape(1, -1)), axis=-1)
            hidden_input = self.fc_hidden.forward(combined_input)
            self.hidden_state = self.tanh.forward(hidden_input)
            self.hidden_states.append(self.hidden_state)

            output_t = self.sigmoid.forward(self.fc_output.forward(self.hidden_state))
            output.append(output_t.flatten())

        return np.array(output)

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros_like(self.fc_hidden.weights)
        self._gradient_weights_output = np.zeros_like(self.fc_output.weights)
        output_error = np.zeros((self.input_tensor.shape[0], self.input_size))
        error = np.zeros((1, self.hidden_size))

        for t in reversed(range(error_tensor.shape[0])):
            input_t = self.inputs[t]
            hidden_state = self.hidden_states[t].reshape(1, -1)
            combined_input = np.concatenate((input_t, hidden_state), axis=-1)
            var = self.tanh.forward(self.fc_hidden.forward(combined_input))
            self.sigmoid.forward(self.fc_output.forward(var))

            output_t = error_tensor[t]
            gradients = self.sigmoid.backward(output_t)
            gradients = self.fc_output.backward(gradients) + error
            self._gradient_weights_output += self.fc_output.gradient_weights

            gradients = self.fc_hidden.backward(self.tanh.backward(gradients))
            self._gradient_weights += self.fc_hidden.gradient_weights

            output_error[t, :] = gradients[:, :self.input_size]
            error = gradients[:, self.input_size:]

        if self._optimizer:
            self.fc_hidden.weights = self._optimizer.calculate_update(self.fc_hidden.weights, self._gradient_weights)
            self.fc_output.weights = self._optimizer.calculate_update(self.fc_output.weights, self._gradient_weights_output)

        return output_error

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fc_hidden._gradient_weights = value

        