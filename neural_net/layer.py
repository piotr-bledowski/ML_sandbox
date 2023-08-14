import numpy as np
from typing import Callable, Any


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
            n (int): output dimension (can be interpreted as number of neurons / linear units in the layer)
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
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward_pass(self, output_error: np.float64, learning_rate: np.float64):
        """
        Computes dL/dA_l-1 given dL/dZ_l as per MIT OCW 6.036 notation

        Parameters:
            output_error (np.float64): output error calculated by the following layer
            learning_rate (np.float64)

        Returns:
            input_error (np.float64): output error of the previous layer
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Bias error is the same as output error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation: Callable[[Any], Any], activation_derivative: Callable[[Any], Any]):
        self.activation = activation
        self.d_activation = activation_derivative

    def forward_pass(self, x: np.array):
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward_pass(self, output_error: np.float64, learning_rate: np.float64):
        """
        Computes dL/dZ_l given dL/dA_l as per MIT OCW 6.036 notation

        Parameters:
            output_error (np.float64): output error calculated by the following layer
            learning_rate (np.float64): (not used in the activation layer because there are no parameters to update there)

        Returns:
            input_error (np.float64): output error of the previous layer
        """
        return self.d_activation(self.input) * output_error
