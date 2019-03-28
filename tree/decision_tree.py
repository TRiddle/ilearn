import copy
import math
import numpy as np

from ..base import BaseClassifier
from .stump import GiniStump
from ..utils.predict import predict_by_mode


class DecisionTree(BaseClassifier):

    def __init__(self, max_depth=None, min_leaf_cnt=None,
                 class_weight=None, splitter='gini'):
        super(DecisionTree, self).__init__()
        self.tag_list_ = [
            'stumps', 'val', 'lch', 'rch'
        ]
        self.decisions_ = []
        self.max_depth_ = max_depth
        self.min_leaf_cnt_ = min_leaf_cnt
        self.class_weight_ = class_weight
        self.splitter_ = splitter
        self.node_cnt_ = 0
        self.EPS_ = 1e-10

    def _init_param(self):
        for tag in self.tag_list_:
            self.param_[tag] = []

    def _create_stump(self):
        if self.splitter_ == 'gini':
            return GiniStump()
        else:
            info = '\"{}\" is an invalid splitter.'.format(self.splitter_)
            raise ValueError(info)

    def _new_idx(self):
        self.node_cnt_ += 1
        return self.node_cnt_ - 1

    def _dfs_fit(self, node_idx, depth, val, node_imp, rows, X, y):
        for tag in self.tag_list_:
            self.param_[tag].append(None)
        self.param_['val'][node_idx] = val
        # same y, pure
        if math.fabs(node_imp) < self.EPS_:
            return
        if depth >= self.max_depth_ or len(rows) <= self.min_leaf_cnt_:
            return
        # try to split
        s = self._create_stump()
        s.fit(X, y, rows)
        # same x, have not stumps
        if s.isnull():
            return
        self.param_['stumps'][node_idx] = s
        lch = self._new_idx()
        self.param_['lch'][node_idx] = lch
        self._dfs_fit(lch, depth+1, s.param_['lch_val'],
                      s.param_['lch_score'], s.param_['lch_rows'], X, y)
        rch = self._new_idx()
        self.param_['rch'][node_idx] = rch
        self._dfs_fit(rch, depth+1, s.param_['rch_val'],
                      s.param_['rch_score'], s.param_['rch_rows'], X, y)

    def _fit(self, X, y):
        self._init_param()
        rows = [r for r in range(X.shape[0])]
        val = predict_by_mode(y, rows)
        self._dfs_fit(self._new_idx(), 0, val, 1, rows, X, y)
        return self

    def _predict(self, X):
        y_hat = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            node_idx = 0
            val = self.param_['val'][node_idx]
            while node_idx < self.node_cnt_:
                s = self.param_['stumps'][node_idx]
                if s is None:
                    break
                if s._data_in_left(X[i, :]):
                    node_idx = self.param_['lch'][node_idx]
                else:
                    node_idx = self.param_['rch'][node_idx]
                val = self.param_['val'][node_idx]
            y_hat[i, 0] = val
        return y_hat

    def _dfs_set_decisions(self, node_idx, decision):
        s = self.param_['stumps'][node_idx]
        if s is None:
            decision['val'] = self.param_['val'][node_idx]
            self.decisions_.append(copy.deepcopy(decision))
            return
        feat = s.param_['feat']
        thres = s.param_['thres']
        lch = self.param_['lch'][node_idx]
        rch = self.param_['rch'][node_idx]
        decision['path'].append((feat, '<=', thres))
        self._dfs_set_decisions(lch, decision)
        decision['path'].pop()
        decision['path'].append((feat, '>', thres))
        self._dfs_set_decisions(rch, decision)
        decision['path'].pop()

    def _set_decisions(self):
        decision = {'path': [], 'val': None}
        self._dfs_set_decisions(0, decision)

    def get_decisions(self):
        if len(self.decisions_) == 0:
            self._set_decisions()
        return self.decisions_
