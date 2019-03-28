import copy
import numpy as np

from ..base import GDBinaryClassifier
from ..utils.activation import sigmoid, sigmoid_grad
from ..utils.cost_function import cross_entropy


class NeuralNetwork(GDBinaryClassifier):

    def __init__(self, hidden_layer_size=[10], activation='sigmoid', alpha=0.1,
                 reg_lambda=1, max_iter=100):
        super(NeuralNetwork, self).__init__(alpha, max_iter)
        self.hidden_layer_size_ = hidden_layer_size
        self.activation_ = activation
        self.lambda_ = reg_lambda
        if activation == 'sigmoid':
            self.act_func = sigmoid
            self.act_grad = sigmoid_grad
        else:
            info = '\"{}\" is an invalid activation.'.format(self.activation_)
            raise ValueError(info)

    def _init_param(self):
        self.layer_size_ = [self.n] + self.hidden_layer_size_ + [1]
        self.nlayers_ = len(self.layer_size_)
        for i in range(1, self.nlayers_):
            W = np.random.randn(self.layer_size_[i], self.layer_size_[i - 1])
            self.param_['W' + str(i)] = W
            b = np.random.randn(self.layer_size_[i], 1)
            self.param_['b' + str(i)] = b

    def _get_pred(self, X):
        A = copy.deepcopy(X)
        self.A_cache_ = [A]
        self.Z_cache_ = [0]
        for i in range(1, self.nlayers_):
            w_i = self.param_['W' + str(i)]
            b_i = self.param_['b' + str(i)]
            Z = np.dot(w_i, A) + b_i
            self.Z_cache_.append(Z)
            if i == self.nlayers_ - 1:
                A = sigmoid(Z)
            else:
                A = self.act_func(Z)
            self.A_cache_.append(A)
        return A

    def _get_cost(self, X, y, y_hat):
        return cross_entropy(y, y_hat)

    def _get_grad(self, X, y, y_hat):
        grad = {}
        dz = y_hat - y
        I = np.ones((self.m, 1))
        for i in range(self.nlayers_ - 1, 0, -1):
            w_cur = self.param_['W' + str(i)]
            a_last = self.A_cache_[i-1]
            z_last = self.Z_cache_[i-1]
            dw = np.dot(dz, a_last.T)
            db = np.dot(dz, I)
            grad['W' + str(i)] = 1.0 * dw / self.m
            grad['b' + str(i)] = 1.0 * db / self.m
            dz = np.multiply(np.dot(w_cur.T, dz), self.act_grad(z_last))
        return grad
