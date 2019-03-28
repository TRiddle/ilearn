import numpy as np


def predict_by_mode(y, rows=None):
    if rows is None:
        rows = [r for r in range(y.shape[0])]
    return np.argmax(np.bincount(y[rows, 0]))

def predict_by_ratio(y, rows=None):
    if rows is None:
        rows = [r for r in range(y.shape[0])]
    bcnt = np.bincount(y[rows, 0])
    if len(bcnt) == 1:
        return 0.0
    else:
        return 1.0 * bcnt[1] / (bcnt[0] + bcnt[1])
