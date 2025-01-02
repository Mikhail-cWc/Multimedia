import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def predict_proba(self, X):
        X = np.array(X)
        return np.array([self._traverse_proba(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1:
            return self._create_leaf(y)

        best_split = self._find_best_split(X, y)
        if not best_split:
            return self._create_leaf(y)

        left_indices, right_indices = best_split['indices']
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_split = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'indices': (left_indices, right_indices)
                    }

        return best_split

    def _gini(self, left, right):
        def gini_impurity(y):
            classes, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return 1 - np.sum(prob ** 2)

        left_gini = gini_impurity(left) if len(left) > 0 else 0
        right_gini = gini_impurity(right) if len(right) > 0 else 0

        return (len(left) * left_gini + len(right) * right_gini) / (len(left) + len(right))

    def _create_leaf(self, y):
        if y.dtype.kind in {'i', 'u'}:  # Классификация
            classes, counts = np.unique(y, return_counts=True)
            probabilities = counts / counts.sum()
            return {'label': classes[np.argmax(probabilities)], 'probabilities': dict(zip(classes, probabilities))}
        else:  # Регрессия
            return {'label': np.mean(y)}

    def _traverse_tree(self, x, tree):
        if 'label' in tree:
            return tree['label']

        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self._traverse_tree(x, tree['left'])
        else:
            return self._traverse_tree(x, tree['right'])

    def _traverse_proba(self, x, tree):
        if 'probabilities' in tree:
            return tree['probabilities']
        feature = tree['feature']
        threshold = tree['threshold']
        if x[feature] <= threshold:
            return self._traverse_proba(x, tree['left'])
        else:
            return self._traverse_proba(x, tree['right'])
