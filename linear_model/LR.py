import numpy as np

from ..base import GDBinaryClassifier
from ..utils.activation import sigmoid, sigmoid_grad
from ..utils.cost_function import cross_entropy


class LogisticRegression(GDBinaryClassifier):

    def __init__(self, alpha=0.1, reg_lambda=1, max_iter=100):
        super(LogisticRegression, self).__init__(alpha, max_iter)
        self.lambda_ = reg_lambda

    def _init_param(self):
        self.param_['W'] = np.random.randn(1, self.n)
        self.param_['b'] = np.random.randn(1, 1)

    def _get_pred(self, X):
        z = np.dot(self.param_['W'], X) + self.param_['b']
        return sigmoid(z)

    def _get_cost(self, X, y, y_hat):
        reg_cost = np.squeeze(np.dot(self.param_['W'], self.param_['W'].T))
        reg_cost = 1.0 * self.lambda_ * reg_cost / self.m / 2
        return cross_entropy(y, y_hat) + reg_cost

    def _get_grad(self, X, y, y_hat):
        I = np.ones((self.m, 1))
        dz = y_hat - y
        dw = np.dot(dz, X.T) + self.lambda_ * self.param_['W']
        db = np.dot(dz, I)
        return {'W': 1.0 * dw / self.m, 'b': 1.0 * db / self.m}
