from abc import ABCMeta, abstractmethod
import numpy as np


class BaseClassifier(object):
    """ Base class of all classifiers
        Param
        -----
        param_: dict
        is_fit_: bool
        num_feat_: int
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.param_ = {}

    def check_data(self, X, y):
        """ Check type and shape for X and y
        """
        if not isinstance(X, np.ndarray):
            raise TypeError('X is not a numpy array.')
        if not isinstance(y, np.ndarray):
            raise TypeError('y is not a numpy array.')
        if len(X.shape) < 2:
            raise ValueError('X is not a matrix.')
        y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError('X.shape do not match y.shape.')
        return X, y

    def check_predict(self, X):
        """ Check type and shape for X, and check fit state
        """
        if not self.is_fit_:
            raise ValueError('Classifier has not been fit.')
        if not isinstance(X, np.ndarray):
            raise TypeError('X is not a numpy array.')
        # after calling check_data, the shape of X is right
        if X.shape[1] != self.num_feat_:
            raise TypeError('The feature number of X is wrong.')

    def accuracy(self, X, y):
        X, y = self.check_data(X, y)
        m = X.shape[0]
        y_pred = self.predict(X)
        acc = 100.0 * np.sum(y_pred == y) / m
        class_name = str(self.__class__).split('.')[-1].strip('.>\'')
        return "Accuracy of {} classifier: {}%".format(class_name, acc)

    def fit(self, X, y, **args):
        """ Check data and train this classifier
            Note
            ----
            You can use as many arguments as you want,
            but don not use positional arguments
        """
        X, y = self.check_data(X, y)
        self.num_feat_ = X.shape[1]
        self.is_fit_ = True
        return self._fit(X, y, args=args)

    def predict(self, X):
        self.check_predict(X)
        m = X.shape[0]
        prob = self.predict_prob(X)
        return (prob > 0.5).reshape(m, -1)

    @abstractmethod
    def _fit(self, X, y, **args):
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

    def _fit(self, X, y, **args):
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
