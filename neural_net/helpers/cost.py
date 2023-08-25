import numpy as np

def MSE(y_pred: np.array, y_actual: np.array) -> np.float64:
    y = y_pred - y_actual
    return np.mean(np.power(y, 2))

def multiclassCrossEntropy(y_pred: np.array, y_actual: np.array) -> np.float64:
    return np.sum(-y_actual * np.log(y_pred))
