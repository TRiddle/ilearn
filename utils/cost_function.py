import numpy as np


def cross_entropy(y, y_hat):
    m = y.shape[1]
    a = np.multiply(y, np.log(y_hat))
    b = np.multiply(1 - y, np.log(1 - y_hat))
    return - 1.0 * np.sum(a + b) / m
