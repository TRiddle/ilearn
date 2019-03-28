import numpy as np


def gini_impurity(y):
    m = y.shape[0]
    cnt = np.sum(y)
    a = 1.0 * cnt / m
    b = 1.0 * (m - cnt) / m
    return 1 - a * a - b * b
