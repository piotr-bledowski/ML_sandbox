import matplotlib.pyplot as plt
import numpy as np

def plotBinaryPoints(X: np.ndarray, y: np.array):
    x_pos = []
    x_neg = []

    for i in range(y.shape[0]):
        if int(y[i]) == 1:
            x_pos.append(X[i])
        else:
            x_neg.append(X[i])

    plt.scatter(np.array(x_pos)[:, 0], np.array(x_pos)[:, 1], color='b')
    plt.scatter(np.array(x_neg)[:, 0], np.array(x_neg)[:, 1], color='r')


def plotLine(theta: np.array, intercept: np.array, xRange: np.array, color: str = 'black', linestyle: str = '-'):
    """ Plot a (separating) line given the normal vector (theta) and point of intercept with the vertical axis """
    y = -(theta[0] / theta[1]) * (xRange - intercept[0]) + intercept[1]
    plt.plot(xRange, y, color=color, linestyle=linestyle)
