import numpy as np

def d_MSE(y_pred: np.array, y_actual: np.array) -> np.float64:
    return 2.0*(y_pred - y_actual) / y_actual.size
