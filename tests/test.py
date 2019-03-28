import os
import sys
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from ilearn.linear_model import LogisticRegression
from ilearn.neural_network import NeuralNetwork
from ilearn.tree import DecisionTree, GiniStump
from ilearn.utils.make_data import make_linearly_separable_data

X, y = make_linearly_separable_data()
print(X.shape, y.shape)
lr = LogisticRegression(alpha=1, reg_lambda=1, max_iter=100)
lr.fit(X, y)
print(lr.accuracy(X, y))

nn = NeuralNetwork(hidden_layer_size=[8, 4], alpha=1.0)
nn.fit(X, y)
print(nn.accuracy(X, y))

stump = GiniStump()
stump.fit(X, y)
print(stump.accuracy(X, y))

tree = DecisionTree(max_depth=5, min_leaf_cnt=10)
tree.fit(X, y)
print(tree.accuracy(X, y))
# print(tree.get_decisions())
