import numpy as np


def sigmoid(z):
    return np.true_divide(1, 1 + np.exp(-z))

def sigmoid_grad(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))
