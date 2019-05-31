from abc import ABCMeta, abstractmethod
import numpy as np

from ..base import BaseClassifier
from ..utils.impurity import gini_impurity
from ..utils.predict import predict_by_ratio


class BaseStump(BaseClassifier):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseStump, self).__init__()
        self.tag_list_ = [
            # cache
            'lch_rows', 'rch_rows', 'lch_prob', 'rch_prob',
            'lch_score', 'rch_score',
            # useful
            'score', 'feat', 'thres'
        ]
        for tag in self.tag_list_:
            self.param_[tag] = None

    def _split(self, X, y, rows, feat, thres):
        rows1 = np.nonzero(X[:, feat] <= thres)[0]
        rows2 = np.nonzero(X[:, feat] >  thres)[0]
        lch_rows = [row for row in list(rows1) if row in rows]
        rch_rows = [row for row in list(rows2) if row in rows]
        return lch_rows, rch_rows

    def _train(self, X, y, rows):
        for feat in range(X.shape[1]):
            thres_set = set(X[rows, feat])
            thres_set.remove(max(thres_set))
            for thres in thres_set:
                lch_rows, rch_rows = self._split(X, y, rows, feat, thres)
                score, lch_score, rch_score = self._score(y, lch_rows, rch_rows,
                                                          feat, thres)
                if score >= self.param_['score']:
                    continue
                self.param_['lch_rows'] = lch_rows
                self.param_['rch_rows'] = rch_rows
                self.param_['lch_score'] = lch_score
                self.param_['rch_score'] = rch_score
                self.param_['score'] = score
                self.param_['feat'] = feat
                self.param_['thres'] = thres
        self.param_['lch_prob'] = predict_by_ratio(y, self.param_['lch_rows'])
        self.param_['rch_prob'] = predict_by_ratio(y, self.param_['rch_rows'])

    def _fit(self, X, y, **args):
        rows = None
        if 'rows' not in args:
            rows = [r for r in range(X.shape[0])]
        else:
            rows = args['rows']
        self._train(X, y, rows)
        return self

    def predict_prob(self, X):
        y_hat = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if self._data_in_left(X[i, :]):
                y_hat[i, 0] = self.param_['lch_prob']
            else:
                y_hat[i, 0] = self.param_['rch_prob']
        return y_hat

    def _data_in_left(self, x):
        if x[self.param_['feat']] <= self.param_['thres']:
            return True
        else:
            return False

    @abstractmethod
    def _score(self, y, lch_rows, rch_rows, feat, thres):
        pass

    @abstractmethod
    def isnull(self):
        pass


class GiniStump(BaseStump):

    def __init__(self):
        super(GiniStump, self).__init__()
        self.IMP_INF = 1e15
        self.param_['score'] = self.IMP_INF

    def _score(self, y, lch_rows, rch_rows, feat, thres):
        lch_cnt = len(lch_rows)
        rch_cnt = len(rch_rows)
        lch_score = gini_impurity(y[lch_rows, :])
        rch_score = gini_impurity(y[rch_rows, :])
        return lch_cnt * lch_score + rch_cnt * rch_score, lch_score, rch_score

    def isnull(self):
        if self.param_['feat'] is None:
            return True
        else:
            return False
