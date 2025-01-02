import numpy as np
from .decision_tree import DecisionTree
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, random_state=None, task="classification"):
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.task = task
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            X_sample, y_sample = resample(X, y, random_state=self.random_state + i)

            if self.max_features is None:
                self.max_features = int(
                    np.sqrt(X.shape[1]) if self.task == "classification" else X.shape[1])
            feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees
        ])

        if self.task == "classification":
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        elif self.task == "regression":
            return np.mean(tree_predictions, axis=0)
