import numpy as np
from numpy import e, typing


def identity(z: np.float64) -> np.float64:
    return z

def ReLU(z: np.float64) -> np.float64:
    return np.max(0.0, z)

def sigmoid(z: np.float64) -> np.float64:
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z: np.float64) -> np.float64:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(z: np.array) -> np.array:
    r = np.array([np.exp(z_i) for z_i in z])
    return r / np.sum(r)
