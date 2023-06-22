from classification.data_generator import generateLinearData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

generateLinearData('data', True)

df = pd.read_csv('data.csv')

X = df[['x1', 'x2']]
y = df['y']

x_pos = []
x_neg = []

for i in range(y.shape[0]):
    if int(y[i]) == 1:
        x_pos.append(X.iloc[i])
    else:
        x_neg.append(X.iloc[i])

x_neg = np.array([[x['x1'], x['x2']] for x in x_neg])
x_pos = np.array([[x['x1'], x['x2']] for x in x_pos])

plt.scatter(x_pos[:,0], x_pos[:,1], color='b')
plt.scatter(x_neg[:,0], x_neg[:,1], color='r')

plt.show()