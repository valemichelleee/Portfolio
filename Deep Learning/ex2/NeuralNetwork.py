import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        inputs = self.input_tensor
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return self.loss_layer.forward(inputs, self.label_tensor)

    def backward(self):
        gradient_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            gradient_tensor = layer.backward(gradient_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for x in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
    def test(self, input_tensor):
        inputs = input_tensor
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
