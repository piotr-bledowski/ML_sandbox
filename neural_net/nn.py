from helpers.activation import *
from typing import Callable, Any
import numpy as np


class Neuron:
    def __init__(self, weights: np.array[np.float64], bias: np.float64, activation: Callable[[Any], Any]):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def __call__(self, x: np.array) -> np.float64:
        return self.activation(np.dot(self.weights, x) + self.bias)

    def update(self, w: np.array[np.float64], b: np.float64):
        self.weights = w
        self.bias = b


class NeuralNetwork:
    pass

