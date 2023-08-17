import numpy as np

def MSE(y_pred: np.array, y_actual: np.array) -> np.float64:
    y = y_pred - y_actual
    return np.sum(np.dot(y.T, y)) / len(y)

def multiclassCrossEntropy(y_pred: np.array, y_actual: np.array) -> np.float64:
    return np.sum(-y_actual * np.log(y_pred))
