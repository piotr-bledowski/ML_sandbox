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
    def __init__(self, layers: list[type[Layer]] = [], loss: Callable[[Any, Any], Any] = MSE, d_loss: Callable[[Any, Any], Any] = d_MSE):
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
        self.training_errors = []

    def addLayer(self, layer: type[Layer]):
        """
        Add layer to the network

        Parameters:
            layer (type[Layer]): new layer

        Returns:
            None
        """
        self.layers.append(layer)

    def batchGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int,
                             learning_rate: float = 0.001,
                             adaptive_step_size_method: str = '',
                             regularization_method: str = ''):
        """
        Batch gradient descent - optimizes on the entire dataset

        Parameters:

        Returns:
            None
        """
        n_samples = len(train_X)

        # keeping track of total mean error in each epoch to use it for plots and early stopping
        epoch_errors = []

        for i in range(n_epochs):
            epoch_error = 0.0
            for j in range(n_samples):
                output = train_X[j]
                # Forward propagation - making a prediction
                for layer in self.layers:
                    output = layer.forward_pass(output)

                # summing epoch error with each datapoint
                epoch_error += self.loss(train_y[j], output)

                # Error back propagation - computing partial derivatives corresponding to each layer
                error = self.d_loss(train_y[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)

            epoch_errors.append(epoch_error / n_samples)

        self.training_errors = epoch_errors


    def miniBatchGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int, batch_size: int,
                                 adaptive_step_size_method: str = '',
                                 regularization_method: str = ''):
        """
        Mini-batch gradient descent - optimizes on a randomly selected part of the data set per iteration

        Parameters:

        Returns:
            None
        """
        pass

    def stochasticGradientDescent(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int,
                                  adaptive_step_size_method: str = '',
                                  regularization_method: str = ''):
        """
        Batch gradient descent - optimizes on only one randomly selected datapoint

        Parameters:

        Returns:
            None
        """
        pass

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, n_epochs: int, algorithm: str,
            adaptive_step_size_method: str = '',
            regularization_method: str = '',
            batch_size: int = 1,
            learning_rate: float = 0.001):
        pass

    def predict(self, x: np.array) -> np.array:
        pass
