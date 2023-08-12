import numpy as np
from activation import sigmoid, tanh

def d_identity(z: np.float64) -> float:
    return 1.0

def d_ReLU(z: np.float64) -> float:
    return 0 if z <= 0 else 1

def d_sigmoid(z: np.float64) -> float:
    return sigmoid(z) * (1.0 - sigmoid(z))

def d_tanh(z: np.float64) -> float:
    return 1.0 - tanh(z)**2

# def d_softmax(z: np.array[np.float64]) -> np.array:
#