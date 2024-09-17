import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.epsilon = np.finfo(np.float64).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        probabilities = np.sum(np.multiply(prediction_tensor, label_tensor), axis=1)
        loss = np.sum(-np.log(probabilities + self.epsilon))
        return loss

    def backward(self, label_tensor):
        return -(label_tensor / (self.prediction_tensor + self.epsilon))