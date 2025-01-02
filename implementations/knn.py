import numpy as np
from collections import Counter


class MyKNN:
    def __init__(self, n_neighbors=3, problem_type='classification'):
        self.n_neighbors = n_neighbors
        self.problem_type = problem_type
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _get_neighbors(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        return np.argsort(distances)[:self.n_neighbors]

    def predict(self, X):
        X = np.atleast_2d(X)
        preds = []
        for x in X:
            neighbors_idx = self._get_neighbors(x)
            neighbors_y = self.y_train[neighbors_idx]

            if self.problem_type == 'classification':
                label_counts = Counter(neighbors_y)
                pred_label = label_counts.most_common(1)[0][0]
                preds.append(pred_label)
            elif self.problem_type == 'regression':
                pred_value = np.mean(neighbors_y)
                preds.append(pred_value)
            else:
                raise ValueError("problem_type должен быть 'classification' или 'regression'.")

        return np.array(preds)
