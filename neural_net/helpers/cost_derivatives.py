import numpy as np

def d_MSE(y_pred: np.array, y_actual: np.array) -> np.array:
    return 2.0*(y_pred - y_actual) / y_actual.size

def d_multiclassCrossEntropy(y_pred: np.array, y_actual: np.array) -> np.array:
    return np.sum(-y_actual / y_pred)
