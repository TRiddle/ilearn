from abc import ABCMeta, abstractmethod
import numpy as np


class BaseClassifier(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.param_ = {}

    def accuracy(self, X, y):
        m, n = X.shape
        y_pred = self.predict(X)
        y_tmp = y.reshape((-1, 1))
        rate = 100.0 * np.sum(y_pred == y_tmp) / m
        class_name = str(self.__class__).split('.')[-1].strip('.>\'')
        return "Accuracy of {} classifier: {}%".format(class_name, rate)

    def check_data(self, X, y):
        if not isinstance(X, np.ndarray):
            raise TypeError('X is not a numpy array.')
        if not isinstance(y, np.ndarray):
            raise TypeError('y is not a numpy array.')
        y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError('X.shape do not match y.shape .')
        return X, y

    def fit(self, X, y):
        X, y = self.check_data(X, y)
        return self._fit(X, y)

    def predict(self, X):
        m, n = X.shape
        prob = self.predict_prob(X)
        return (prob > 0.5).reshape(m, -1)

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def predict_prob(self, X):
        pass


class BaseOptimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, alpha):
        self.alpha_ = alpha

    @abstractmethod
    def update(self, param, grad):
        pass


class BGDOptimizer(BaseOptimizer):

    def __init__(self, alpha):
        super(BGDOptimizer, self).__init__(alpha)

    def update(self, param, grad):
        for p in param:
            param[p] -= self.alpha_ * grad[p]


class GDBinaryClassifier(BaseClassifier):
    __metaclass__ = ABCMeta

    def __init__(self, alpha, max_iter, optimizer='bgd'):
        super(GDBinaryClassifier, self).__init__()
        self.max_iter_ = max_iter
        if optimizer == 'bgd':
            self.optimizer_ = BGDOptimizer(alpha)
        else:
            raise ValueError('\"{}\" is an invalid optimizer.'.format(optimizer))

    def _train(self, X, y):
        cost_list = []
        for i in range(self.max_iter_):
            y_hat = self._get_pred(X)
            cost_list.append(self._get_cost(X, y, y_hat))
            grad = self._get_grad(X, y, y_hat)
            self.optimizer_.update(self.param_, grad)
        return cost_list

    def _fit(self, X, y):
        self.m, self.n = X.shape
        self._init_param()
        self.cost_history_ = self._train(X.T, y.T)
        return self

    def predict_prob(self, X):
        y_hat = self._get_pred(X.T)
        return y_hat.T

    @abstractmethod
    def _init_param(self, X, y):
        pass

    @abstractmethod
    def _get_pred(self, X):
        pass

    @abstractmethod
    def _get_cost(self, X, y, y_hat):
        pass

    @abstractmethod
    def _get_grad(self, X, y, y_hat):
        pass
