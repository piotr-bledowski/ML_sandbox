from helpers.activation import *  # already imports numpy as np
from helpers.cost import MSE
from helpers.cost_derivatives import d_MSE
from layer import *
import random


# class Neuron:
#     def __init__(self, dim: int, activation: Callable[[Any], Any]):
#         self.weights = np.random.rand(dim)
#         self.bias = random.random()
#         self.activation = activation
#
#     def __call__(self, x: np.array) -> np.float64:
#         return self.activation(np.dot(self.weights.T, x) + self.bias)
#
#     def update(self, w: np.array[np.float64], b: np.float64):
#         self.weights = w
#         self.bias = b


class NeuralNetwork:
    def __init__(self, layers: list[type[Layer]] = None, loss: Callable[[Any], Any] = MSE, d_loss: Callable[[Any], Any] = d_MSE):
        """
        Sequential Neural Net constructor

        Parameters: (all of them optional)
            layers (list[type[Layer]]): initialize with layers
            loss (Callable[[Any], Any]): initialize with a loss function (default - MSE)
            d_loss (Callable[[Any], Any]): will be derived automatically from the provided loss function

        Returns:
            None
        """
        self.layers = layers
        self.loss = loss
        self.d_loss = d_loss


