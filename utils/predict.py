import numpy as np


def predict_by_mode(y, rows=None):
    if rows is None:
        rows = [r for r in range(y.shape[0])]
    return np.argmax(np.bincount(y[rows, 0]))
