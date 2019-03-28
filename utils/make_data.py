from sklearn.datasets import make_classification
import numpy as np


def make_linearly_separable_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    X += 2 * np.random.RandomState(2).uniform(size=X.shape)
    return X, y
