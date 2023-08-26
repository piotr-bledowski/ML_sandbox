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
from neural_net.layer import FullyConnectedLayer, ActivationLayer
from neural_net.helpers.activation import *
from neural_net.helpers.activation_derivatives import *
from neural_net.helpers.cost import *
from neural_net.helpers.cost_derivatives import *

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


train_data = pd.read_csv('mnist_train.csv')

X_train = train_data.loc[:, train_data.columns != 'label'].to_numpy(dtype=np.float64)
# print(X_train.dtype)
X_train /= 255.0  # normalize
y_train = np.array(to_categorical(train_data['label']), dtype=np.float64)

X_train = np.expand_dims(X_train, axis=1)

# Split into training and validation data
X_valid = X_train[59000:]
y_valid = y_train[59000:]
X_train = X_train[:59000]
y_train = y_train[:59000]

# print(X_train.shape)

# print(X_train)
# print(y_train)

model = NeuralNetwork(layers=[
    FullyConnectedLayer(784, 200),
    ActivationLayer(sigmoid, d_sigmoid),
    FullyConnectedLayer(200, 100),
    ActivationLayer(sigmoid, d_sigmoid),
    FullyConnectedLayer(100, 50),
    ActivationLayer(sigmoid, d_sigmoid),
    FullyConnectedLayer(50, 10),
    ActivationLayer(sigmoid, d_sigmoid)
])

# model.fit(X_train, y_train, X_valid, y_valid, n_epochs=1000, algorithm='mbgd', batch_size=100, learning_rate=0.1)
model.fit(X_train, y_train, X_valid, y_valid, n_epochs=300000, algorithm='sgd', learning_rate=0.01)

test_data = pd.read_csv('mnist_test.csv')

X_test = test_data.loc[:, train_data.columns != 'label'].to_numpy(dtype=np.float64)
X_test /= 255.0  # normalize
X_test = np.expand_dims(X_test, axis=1)
y_test = test_data['label']

predictions = np.array([np.argmax(p) for p in model.predict(X_test)])

right = np.count_nonzero(predictions == y_test)
wrong = len(y_test) - right

print(f'{100.0*float(right)/float((right + wrong))}% accuracy')
