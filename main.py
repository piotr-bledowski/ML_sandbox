import sys
import os

sys.path.append(f'{os.getcwd()}/neural_net')
sys.path.append(f'{os.getcwd()}/neural_net/helpers')

from classification.data_generator import generateLinearData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from classification.models.perceptron import Perceptron
from plotting import plotBinaryPoints, plotLine
from neural_net.nn import NeuralNetwork

from keras.utils import to_categorical

# generateLinearData('data', True, n_samples=10)
#
# df = pd.read_csv('data.csv')
#
# X = df[['x1', 'x2']].to_numpy()
# y = df['y'].to_numpy()
#
# plotBinaryPoints(X, y)
#
# model = Perceptron()
#
# model.fit(X, y, 100000)
#
# xRange = np.array([min(X[0]) - 1, max(X[0]) + 1])
#
# plotLine(model.theta, np.array([0, model.theta_0]), xRange)
#
# print(model.theta)
# print(model.theta_0)
#
# plt.show()


train_data = pd.read_csv('mnist_test.csv')

y_train = to_categorical(train_data['label'])

print(y_train)

