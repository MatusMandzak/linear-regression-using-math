import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


plt.rcParams['figure.figsize'] = (12.0, 9.0)# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
X = X.to_numpy().reshape(-1,1)
X = np.c_[np.ones(len(X)),X]
Y = Y.to_numpy().reshape(-1,1)
Y = np.array(Y)
inp = X[:, [1]]

m = len(X)
theta = np.array([[0.],[0.]])

L = 0.001  # The learning Rate
epochs = 5000  # The number of iterations to perform gradient descent

# Performing Batch Gradient Descent 
for i in range(epochs):
    for j in range(len(theta)):
        s = 0
        for k in range(m):
            hx = np.sum(theta.T*X[k],axis=1)
            s += (hx-Y[k][0])*X[k,j]
        theta[j][0] = theta[j][0] - L*s

print(theta)

Y_pred = theta[1]*inp + theta[0]
plt.scatter(inp, Y)
plt.plot([min(inp), max(inp)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

#stochaistic gradient descent
theta = np.array([[0.],[0.]])
for i in range(epochs):
    for k in range(m):
        for j in range(len(theta)):
            hx = np.sum(theta.T*X[k],axis=1)
            s = (hx-Y[k])*X[k,j]
            theta[j][0] = theta[j][0] - L*s

print(theta)


Y_pred = theta[1]*inp + theta[0]
plt.scatter(inp, Y)
plt.plot([min(inp), max(inp)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()
