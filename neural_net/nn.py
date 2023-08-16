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
    def __init__(self, layers: list[type[Layer]] = [], loss: Callable[[Any], Any] = MSE, d_loss: Callable[[Any], Any] = d_MSE):
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

    def addLayer(self, layer: type[Layer]):
        """
        Add layer to the network

        Parameters:
            layer (type[Layer]): new layer

        Returns:
            None
        """
        self.layers.append(layer)

    def batchGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int):
        pass

    def miniBatchGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int, batch_size: int):
        pass

    def stochasticGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int):
        pass

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int, algorithm: str,
            adaptive_step_size_method: str = '',
            regularization_method: str = '',
            batch_size: int = 1,
            learning_rate: float = 0.001):
        pass

    def predict(self, x: np.array) -> np.array:
        pass
