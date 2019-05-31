from ..base import BaseClassifier

import numpy as np


class VotingClassifier(BaseClassifier):

    def __init__(self, clfs, voting):
        self.clfs_ = clfs
        self.voting_ = voting
        if voting not in ['soft', 'hard']:
            raise ValueError('\"{}\" is an invalid voting'.format(voting))

    def _fit(self, X, y, **args):
        for name, clf in self.clfs_:
            clf.fit(X, y)

    def predict_prob(self, X):
        m, n = X.shape
        prob = np.zeros((m, 1))
        for name, clf in self.clfs_:
            if self.voting_ == 'soft':
                prob += clf.predict_prob(X).reshape((m, -1))
            else:
                prob += clf.predict(X).reshape((m, -1))
        prob = np.true_divide(prob, len(self.clfs_))
        return prob
