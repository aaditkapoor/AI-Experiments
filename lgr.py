# Logistic regression from scratch

import numpy as np
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost_function(X, y, weights):
    z = np.dot(X, weights) # x1w1 + x2w2 ... + xnwn
    # two parts of the cost function
    # if true label is 1 or true label is 0
    predict_1 = y * np.log(sigmoid(z))
    predict_0 = (1-y) * np.log(1 - sigmoid(z))

    # combined and reduce sum of the two eqns
    return -sum(predict_1 + predict_0) / len(X)

def train(X, y, epochs=100, lr = 0.03):
    loss = [] # collection of weights
    weights = np.random.rand(X.shape[1]) # (None, features)
    N = len(X)

    for e in range(epochs):
        y_hat = sigmoid(np.dot(X, weights)) # activation of weighted sum

        # updated weights
        # chain rule to get simplified equ
        # loss function derivative and updated using the gradient update eqn
        weights -= lr * np.dot(X.T, y_hat - y) / N
        s = "ep {}".format(e)
        loss.append({s: cost_function(X, y, weights)})

    
    return weights, loss


def predict(X, weights):
    z = np.dot(X, weights)
    return [1 if i > 0.5 else 0 for i in sigmoid(z)]


X,y = make_classification(n_samples=2,n_features=4)

weights, loss = train(X, y)
predict(X, weights)
