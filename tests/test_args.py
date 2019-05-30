from abc import ABCMeta, abstractmethod
import numpy as np


class Test(object):

    __metaclass__ = ABCMeta

    def fit(self, X, y, **args):
        print(args)
        self._fit(X, y, args=args)

    @abstractmethod
    def _fit(self, X, y, **args):
        pass

class SubTest(Test):

    def _fit(self, X, y, **args):
        pass

sub_test = SubTest()
X = np.random.randn(3, 3)
y = np.random.randn(3, 1)
sub_test.fit(X, y, rows=[1, 2, 3])
