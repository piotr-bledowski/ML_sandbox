import sys
import os

import numpy as np

sys.path.append(f'{os.getcwd()}/helpers')

from activation import *  # already imports numpy as np
from cost import MSE
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
    def __init__(self, layers: list[type[Layer]], loss: Callable[[Any, Any], Any] = MSE, d_loss: Callable[[Any, Any], Any] = d_MSE):
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
        self.validation_errors = []

    def addLayer(self, layer: type[Layer]):
        """
        Add layer to the network

        Parameters:
            layer (type[Layer]): new layer

        Returns:
            None
        """
        self.layers.append(layer)

    def batchGradientDescent(self, X_train: np.ndarray, y_train: np.ndarray,
                             n_epochs: int,
                             learning_rate: float,
                             X_valid: np.ndarray = None,
                             y_valid: np.ndarray = None,
                             adaptive_step_size_method: str = '',
                             regularization_method: str = '',
                             gamma: np.float64 = 0.9,
                             epsilon: np.float64 = 0.001,
                             beta_1: np.float64 = 0.9,
                             beta_2: np.float64 = 0.9,
                             t: int = 1):
        """
        Batch gradient descent - optimizes on the entire dataset

        Parameters:
            X_train (np.ndarray): training non-target data
            y_train (np.ndarray): training target data
            n_epochs (int): number of training epochs
            learning_rate (float): learning rate
            X_valid (np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            y_valid(np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            adaptive_step_size_method (str): optional method to improve learning by making step size adaptive (momentum, adadelta, adam)
            regularization_method (str): optional ()
            gamma (np.float64): optional momentum / adadelta constant, 0.9 by default
            epsilon (np.float64): optional adadelta constant, 0.001 by default
            beta_1 (np.float64): optional adam constant (for first moment estimate as per original paper naming)
            beta_2 (np.float64): optional adam constant (for second moment estimate as per original paper naming)
            t (int): optional (only for Adam) epoch number - necessary information for Adam
        Returns:
            None
        """
        n_samples = len(X_train)

        # keeping track of total mean error in each epoch to use it for plots and early stopping
        epoch_errors = []

        if adaptive_step_size_method == 'momentum':
            for i in range(n_epochs):
                epoch_error = 0.0
                for j in range(n_samples):
                    output = X_train[j]
                    # Forward propagation - computing output
                    for layer in self.layers:
                        output = layer.forwardPass(output)

                    # summing epoch error with each datapoint
                    epoch_error += self.loss(output, y_train[j])

                    # Error back propagation - computing partial derivatives corresponding to each layer
                    error = self.d_loss(output, y_train[j])
                    for layer in reversed(self.layers):
                        error = layer.backwardPassMomentum(error, learning_rate, gamma)

                epoch_errors.append(epoch_error / n_samples)
        elif adaptive_step_size_method == 'adadelta':
            for i in range(n_epochs):
                epoch_error = 0.0
                for j in range(n_samples):
                    output = X_train[j]
                    # Forward propagation - computing output
                    for layer in self.layers:
                        output = layer.forwardPass(output)

                    # summing epoch error with each datapoint
                    epoch_error += self.loss(output, y_train[j])

                    # Error back propagation - computing partial derivatives corresponding to each layer
                    error = self.d_loss(output, y_train[j])
                    for layer in reversed(self.layers):
                        error = layer.backwardPassAdadelta(error, learning_rate, gamma, epsilon)

                epoch_errors.append(epoch_error / n_samples)
        elif adaptive_step_size_method == 'adam':
            for i in range(n_epochs):
                epoch_error = 0.0
                for j in range(n_samples):
                    output = X_train[j]
                    # Forward propagation - computing output
                    for layer in self.layers:
                        output = layer.forwardPass(output)

                    # summing epoch error with each datapoint
                    epoch_error += self.loss(output, y_train[j])

                    # Error back propagation - computing partial derivatives corresponding to each layer
                    error = self.d_loss(output, y_train[j])
                    for layer in reversed(self.layers):
                        error = layer.backwardPassAdam(error, learning_rate, epsilon, beta_1, beta_2, t)

                epoch_errors.append(epoch_error / n_samples)
        else:
            # no fancy adaptive step size methods
            for i in range(n_epochs):
                epoch_error = 0.0
                for j in range(n_samples):
                    output = X_train[j]
                    # Forward propagation - computing output
                    for layer in self.layers:
                        output = layer.forwardPass(output)

                    # summing epoch error with each datapoint
                    epoch_error += self.loss(output, y_train[j])

                    # Error back propagation - computing partial derivatives corresponding to each layer
                    error = self.d_loss(output, y_train[j])
                    for layer in reversed(self.layers):
                        error = layer.backwardPass(error, learning_rate)

                epoch_errors.append(epoch_error / n_samples)

        self.training_errors = epoch_errors

    def miniBatchGradientDescent(self, X_train: np.ndarray, y_train: np.ndarray,
                                 n_epochs: int,
                                 learning_rate: float,
                                 batch_size: int,
                                 X_valid: np.ndarray = None,
                                 y_valid: np.ndarray = None,
                                 adaptive_step_size_method: str = '',
                                 regularization_method: str = '',
                                 gamma: np.float64 = 0.9,
                                 epsilon: np.float64 = 0.001,
                                 beta_1: np.float64 = 0.9,
                                 beta_2: np.float64 = 0.9):
        """
        Mini-batch gradient descent - optimizes on a randomly selected part of the data set per iteration

        Parameters:
            X_train (np.ndarray): training non-target data
            y_train (np.ndarray): training target data
            n_epochs (int): number of training epochs
            learning_rate (float): learning rate
            batch_size (int): batch size for mini-batch gradient descent only
            X_valid (np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            y_valid(np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            adaptive_step_size_method (str): optional method to improve learning by making step size adaptive (momentum, adadelta, adam)
            regularization_method (str): optional ()
            gamma (np.float64): optional momentum / adadelta constant, 0.9 by default
            epsilon (np.float64): optional adadelta constant, 0.001 by default
            beta_1 (np.float64): optional adam constant (for first moment estimate as per original paper naming)
            beta_2 (np.float64): optional adam constant (for second moment estimate as per original paper naming)
        Returns:
            None
        """

        # keeping track of total mean error in each epoch to use it for plots and early stopping
        epoch_errors = []

        n_samples = len(X_train)

        for i in range(n_epochs):
            indices = np.random.randint(0, n_samples, size=batch_size)
            self.batchGradientDescent(X_train[indices], y_train[indices], X_valid=X_valid, y_valid=y_valid, n_epochs=1,
                                      learning_rate=learning_rate, adaptive_step_size_method=adaptive_step_size_method,
                                      regularization_method=regularization_method, gamma=gamma, epsilon=epsilon,
                                      beta_1=beta_1, beta_2=beta_2, t=i+1)
            epoch_errors.append(self.training_errors[0])
            print(f'Epoch {i + 1} training error: {self.training_errors[0]}')
            if X_valid is not None and y_valid is not None:
                validation_error = self.validationError(X_valid, y_valid)
                self.validation_errors.append(validation_error)
                print(f'Epoch {i + 1} validation error: {validation_error}')

        self.training_errors = epoch_errors

    def stochasticGradientDescent(self, X_train: np.ndarray, y_train: np.ndarray,
                                  n_epochs: int,
                                  learning_rate: float,
                                  X_valid: np.ndarray = None,
                                  y_valid: np.ndarray = None,
                                  adaptive_step_size_method: str = '',
                                  regularization_method: str = '',
                                  gamma: np.float64 = 0.9,
                                  epsilon: np.float64 = 0.001,
                                  beta_1: np.float64 = 0.9,
                                  beta_2: np.float64 = 0.9):
        """
        Batch gradient descent - optimizes on only one randomly selected datapoint

        Parameters:
            X_train (np.ndarray): training non-target data
            y_train (np.ndarray): training target data
            n_epochs (int): number of training epochs
            learning_rate (float): learning rate
            X_valid (np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            y_valid(np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            adaptive_step_size_method (str): optional method to improve learning by making step size adaptive (momentum, adadelta, adam)
            regularization_method (str): optional ()
            gamma (np.float64): optional momentum / adadelta constant, 0.9 by default
            epsilon (np.float64): optional adadelta constant, 0.001 by default
            beta_1 (np.float64): optional adam constant (for first moment estimate as per original paper naming)
            beta_2 (np.float64): optional adam constant (for second moment estimate as per original paper naming)
        Returns:
            None
        """
        # keeping track of total mean error in each epoch to use it for plots and early stopping
        epoch_errors = []

        n_samples = len(X_train)

        if adaptive_step_size_method == 'momentum':
            for i in range(n_epochs):
                ind = random.randint(0, n_samples - 1)  # pick a random datapoint by index
                x = X_train[ind]
                y = y_train[ind]

                output = x
                # Forward propagation - computing output
                for layer in self.layers:
                    output = layer.forwardPass(output)

                # Error back propagation - computing partial derivatives corresponding to each layer
                error = self.d_loss(output, y)

                epoch_errors.append(self.loss(output, y))

                print(f'Epoch {i + 1} training error: {epoch_errors[-1]}')

                for layer in reversed(self.layers):
                    error = layer.backwardPassMomentum(error, learning_rate, gamma)
        elif adaptive_step_size_method == 'adadelta':
            for i in range(n_epochs):
                ind = random.randint(0, n_samples - 1)  # pick a random datapoint by index
                x = X_train[ind]
                y = y_train[ind]

                output = x
                # Forward propagation - computing output
                for layer in self.layers:
                    output = layer.forwardPass(output)

                # Error back propagation - computing partial derivatives corresponding to each layer
                error = self.d_loss(output, y)

                epoch_errors.append(self.loss(output, y))

                print(f'Epoch {i + 1} training error: {epoch_errors[-1]}')

                for layer in reversed(self.layers):
                    error = layer.backwardPassAdadelta(error, learning_rate, gamma, epsilon)
        elif adaptive_step_size_method == 'adam':
            for i in range(n_epochs):
                ind = random.randint(0, n_samples - 1)  # pick a random datapoint by index
                x = X_train[ind]
                y = y_train[ind]

                output = x
                # Forward propagation - computing output
                for layer in self.layers:
                    output = layer.forwardPass(output)

                # Error back propagation - computing partial derivatives corresponding to each layer
                error = self.d_loss(output, y)

                epoch_errors.append(self.loss(output, y))

                print(f'Epoch {i + 1} training error: {epoch_errors[-1]}')

                for layer in reversed(self.layers):
                    error = layer.backwardPassAdam(error, learning_rate, epsilon, beta_1, beta_2, t=i+1)
        else:
            # no fancy adaptive step size methods
            for i in range(n_epochs):
                ind = random.randint(0, n_samples-1)  # pick a random datapoint by index
                x = X_train[ind]
                y = y_train[ind]

                output = x
                # Forward propagation - computing output
                for layer in self.layers:
                    output = layer.forwardPass(output)

                # Error back propagation - computing partial derivatives corresponding to each layer
                error = self.d_loss(output, y)

                epoch_errors.append(self.loss(output, y))

                print(f'Epoch {i + 1} training error: {epoch_errors[-1]}')

                for layer in reversed(self.layers):
                    error = layer.backwardPass(error, learning_rate)

        self.training_errors = epoch_errors

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_epochs: int,
            learning_rate: float,
            algorithm: str,
            X_valid: np.ndarray = None,
            y_valid: np.ndarray = None,
            adaptive_step_size_method: str = '',
            regularization_method: str = '',
            batch_size: int = 1,
            gamma: np.float64 = 0.9,
            epsilon: np.float64 = 0.001,
            beta_1: np.float64 = 0.9,
            beta_2: np.float64 = 0.9):
        """
        fit the Neural Network to some training data

        Parameters:
            X_train (np.ndarray): training non-target data
            y_train (np.ndarray): training target data
            n_epochs (int): number of training epochs
            learning_rate (float): learning rate
            algorithm (str): type of gradient descent algorithm (bgd, mbgd, sgd)
            X_valid (np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            y_valid(np.ndarray): optional validation data necessary for early stopping and keeping track of validation error along training in general
            adaptive_step_size_method (str): optional method to improve learning by making step size adaptive (momentum, adadelta, adam)
            regularization_method (str): ()
            batch_size (int): batch size required for mini-batch gradient descent only
            gamma (np.float64): optional momentum / adadelta constant, 0.9 by default
            epsilon (np.float64): optional adadelta constant, 0.001 by default
            beta_1 (np.float64): optional adam constant (for first moment estimate as per original paper naming)
            beta_2 (np.float64): optional adam constant (for second moment estimate as per original paper naming)
        Returns:
            None
        """
        if algorithm == 'bgd':
            self.batchGradientDescent(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_epochs=n_epochs, learning_rate=learning_rate,
                                      adaptive_step_size_method=adaptive_step_size_method, regularization_method=regularization_method,
                                      gamma=gamma, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        elif algorithm == 'mbgd':
            self.miniBatchGradientDescent(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_epochs=n_epochs, learning_rate=learning_rate, batch_size=batch_size,
                                          adaptive_step_size_method=adaptive_step_size_method, regularization_method=regularization_method,
                                          gamma=gamma, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        elif algorithm == 'sgd':
            self.stochasticGradientDescent(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_epochs=n_epochs, learning_rate=learning_rate,
                                           adaptive_step_size_method=adaptive_step_size_method, regularization_method=regularization_method,
                                           gamma=gamma, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def predict(self, x: np.array) -> np.array:
        predictions = []

        for i in range(len(x)):
            output = x[i]
            for layer in self.layers:
                output = layer.forwardPass(output)
            predictions.append(np.argmax(output))

        return np.array(predictions)

    def validationError(self, X_valid, y_valid):
        error = 0.0
        for i in range(len(X_valid)):
            output = X_valid[i]
            for layer in self.layers:
                output = layer.forwardPass(output)
            error += self.loss(output, y_valid[i])
        return error / len(y_valid)

    def validatePercent(self, X_valid, y_valid):
        predictions = self.predict(X_valid)
        right = np.count_nonzero(predictions == y_valid)
        wrong = len(y_valid) - right
        return float(right)/float((right + wrong))
