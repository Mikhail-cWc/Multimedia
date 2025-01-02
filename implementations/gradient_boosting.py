import numpy as np

from .decision_tree import DecisionTree

from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin


class GradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.n_classes = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_classes = len(np.unique(y))

        y_one_hot = np.zeros((len(y), self.n_classes))
        y_one_hot[np.arange(len(y)), y] = 1

        F = np.zeros((len(y), self.n_classes))

        for _ in range(self.n_estimators):
            trees = []
            for k in range(self.n_classes):
                residual = y_one_hot[:, k] - self._softmax(F)[:, k]

                tree = DecisionTree(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split)
                tree.fit(X, residual)
                trees.append(tree)

                F[:, k] += self.learning_rate * tree.predict(X)

            self.trees.append(trees)

        return self

    def predict(self, X):
        X = np.array(X)
        F = np.zeros((len(X), self.n_classes))

        for trees in self.trees:
            for k, tree in enumerate(trees):
                F[:, k] += self.learning_rate * tree.predict(X)

        return np.argmax(F, axis=1)

    def predict_proba(self, X):
        X = np.array(X)
        F = np.zeros((len(X), self.n_classes))

        for trees in self.trees:
            for k, tree in enumerate(trees):
                F[:, k] += self.learning_rate * tree.predict(X)

        return self._softmax(F)

    def _softmax(self, F):
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)
