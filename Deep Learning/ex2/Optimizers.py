import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        updated_weights = self.v + weight_tensor
        return updated_weights


class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.g = 0
        self.v = 0
        self.r = 0
        self.k = 0
        self.epsilon = np.finfo(np.float64).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k = self.k + 1
        self.g = gradient_tensor
        self.v = self.mu * self.v + (1 - self.mu) * self.g
        self.r = self.rho * self.r + np.multiply((1 - self.rho) * self.g, self.g)
        v_hat = self.v / (1 - self.mu ** self.k)
        r_hat = self.r / (1 - self.rho ** self.k)
        updated_weights = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)
        return updated_weights
