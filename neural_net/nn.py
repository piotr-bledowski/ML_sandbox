from helpers.activation import *  # already imports numpy as np
from helpers.cost import MSE
from typing import Callable, Any
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


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, x: np.array):
        """
        Computes the output of a layer for a given input
        Z_l or A_l depending on the layer's type as per MIT OCW 6.036 notation
        """
        pass

    def backward_pass(self, output_error: np.float64, learning_rate: np.float64):
        """
        Computes dL/dZ_l given dL/dA_l as per MIT OCW 6.036 notation
        """
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, m: int, n: int):
        """
        Initializes a fully connected NN layer

        Parameters:
            m (int): input dimension
            n (int): output dimension (can be interpreted as number of neurons in the layer)
        """
        self.weights = np.random.rand(m, n)  # m weights for each of n neurons, can be neatly represented as a matrix
        self.bias = np.random.rand(1, n)  # n biases for n neurons

    def forward_pass(self, x: np.array) -> np.array[np.float64]:
        """
        Instead of keeping track of every single neuron, we can multiply the layer's input
        by the layer's weight matrix getting the output vector which we can feed to an activation layer.
        Resulting vector is 1xn, so is self.bias.

        Parameters:
            x (np.array): layer's input

        Returns:
            z (np.array[np.float64]): layer's output
        """
        return np.dot(x, self.weights) + self.bias


class ActivationLayer(Layer):
    pass


class NeuralNetwork:
    pass

