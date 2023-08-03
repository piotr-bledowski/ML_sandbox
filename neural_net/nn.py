from helpers.activation import *  # already imports numpy as np
from typing import Callable, Any


class Neuron():
    def __init__(self, weights: np.array[np.float64], activation: Callable[[Any], Any]):
        self.weights = weights
        self.activation = activation


class NeuralNetwork():
    pass

