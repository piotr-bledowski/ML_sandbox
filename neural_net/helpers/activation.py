import numpy as np
from numpy import e


def identity(z: np.float64) -> np.float64:
    return z

def ReLU(z: np.float64) -> np.float64:
    return np.max(0.0, z)

def sigmoid(z: np.float64) -> np.float64:
    return 1 / (1 + e**(-z))

def tanh(z: np.float64) -> np.float64:
    return (e**z - e**(-z)) / (e**z + e**(-z))