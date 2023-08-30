import numpy as np
from typing import Callable, Any


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forwardPass(self, x: np.array):
        """
        Computes the output of a layer for a given input
        Z_l or A_l depending on the layer's type as per MIT OCW 6.036 notation
        """
        pass

    def backwardPass(self, output_error: np.float64, learning_rate: np.float64):
        """
        Computes dL/dZ_l given dL/dA_l as per MIT OCW 6.036 notation
        """
        pass

    def backwardPassAdadelta(self, output_error: np.float64, learning_rate: np.float64, gamma: np.float64 = 0.9, epsilon: np.float64 = 0.001):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, m: int, n: int):
        """
        Initializes a fully connected NN layer

        Parameters:
            m (int): input dimension
            n (int): output dimension (can be interpreted as number of neurons / linear units in the layer)
        """
        self.weights = np.random.rand(m, n) - 0.5  # m weights for each of n neurons, can be neatly represented as a matrix
        self.bias = np.random.rand(1, n) - 0.5  # n biases for n neurons
        self.weights_momentum = 0
        self.bias_momentum = 0
        self.weights_adadelta = 0
        self.bias_adadelta = 0

    def forwardPass(self, x: np.array) -> np.array:
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

    def backwardPass(self, output_error: np.array, learning_rate: np.float64):
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

    def backwardPassMomentum(self, output_error: np.array, learning_rate: np.float64, gamma: np.float64 = 0.9):
        """
                Computes dL/dA_l-1 given dL/dZ_l as per MIT OCW 6.036 notation

                Parameters:
                    output_error (np.float64): output error calculated by the following layer
                    learning_rate (np.float64)
                    gamma (np.float64): momentum constant

                Returns:
                    input_error (np.float64): output error of the previous layer
                """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Exponential Moving Average implementation for both weights and bias

        self.weights_momentum = gamma * self.weights_momentum + (1 - gamma) * weights_error
        self.bias_momentum = gamma * self.bias_momentum + (1 - gamma) * output_error

        self.weights -= learning_rate * self.weights_momentum
        self.bias -= learning_rate * self.bias_momentum

        return input_error

    def backwardPassAdadelta(self, output_error: np.float64, learning_rate: np.float64, gamma: np.float64 = 0.9, epsilon: np.float64 = 0.001):
        """
        Computes dL/dA_l-1 given dL/dZ_l as per MIT OCW 6.036 notation

        Parameters:
            output_error (np.float64): output error calculated by the following layer
            learning_rate (np.float64)
            gamma (np.float64): momentum constant

        Returns:
            input_error (np.float64): output error of the previous layer
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Exponential Moving Average implementation for both weights and bias

        self.weights_adadelta = gamma * self.weights_adadelta + (1 - gamma) * weights_error**2
        self.bias_adadelta = gamma * self.bias_adadelta + (1 - gamma) * output_error**2

        self.weights -= learning_rate / np.sqrt(self.weights_adadelta + epsilon) * weights_error
        self.bias -= learning_rate / np.sqrt(self.bias_adadelta + epsilon) * output_error

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation: Callable[[Any], Any], activation_derivative: Callable[[Any], Any]):
        self.activation = activation
        self.d_activation = activation_derivative

    def forwardPass(self, x: np.array):
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backwardPass(self, output_error: np.float64, learning_rate: np.float64 = 1):
        """
        Computes dL/dZ_l given dL/dA_l as per MIT OCW 6.036 notation

        Parameters:
            output_error (np.float64): output error calculated by the following layer
            learning_rate (np.float64): (not used in the activation layer because there are no parameters to update there)

        Returns:
            input_error (np.float64): output error of the previous layer
        """
        return self.d_activation(self.input) * output_error

    def backwardPassMomentum(self, output_error: np.array, learning_rate: np.float64, gamma: np.float64 = 0.9):
        return self.backwardPass(output_error, learning_rate)

    def backwardPassAdadelta(self, output_error: np.float64, learning_rate: np.float64, gamma: np.float64 = 0.9, epsilon: np.float64 = 0.001):
        return self.backwardPass(output_error, learning_rate)
