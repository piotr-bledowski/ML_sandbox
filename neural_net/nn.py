from helpers.activation import *  # already imports numpy as np
from typing import Callable


class Neuron():
    def __init__(self, weights: np.array, activation: Callable[[np.float64], np.float64]):
        self.weights = weights
        self.activation = activation


class NeuralNetwork():
    pass

