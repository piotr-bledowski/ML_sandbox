from helpers.activation import *  # already imports numpy as np
from typing import Callable, Any
import random


class Neuron:
    def __init__(self, dim: int, activation: Callable[[Any], Any]):
        self.weights = np.random.rand(dim)
        self.bias = random.random()
        self.activation = activation

    def __call__(self, x: np.array) -> np.float64:
        return self.activation(np.dot(self.weights, x) + self.bias)

    def update(self, w: np.array[np.float64], b: np.float64):
        self.weights = w
        self.bias = b


class NeuralNetwork:
    pass

