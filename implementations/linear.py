import numpy as np


class MyLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None  # будет размером (n_features, n_classes)
        self.bias = None     # будет размером (n_classes,)

    def _one_hot(self, y, n_classes):
        one_hot_matrix = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            one_hot_matrix[i, label] = 1.0
        return one_hot_matrix

    def _softmax(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        y_one_hot = self._one_hot(y, n_classes)

        for _ in range(self.n_iters):
            Z = np.dot(X, self.weights) + self.bias  # [N, K]

            P = self._softmax(Z)  # [N, K]
            dW = (1 / n_samples) * np.dot(X.T, (P - y_one_hot))
            db = (1 / n_samples) * np.sum(P - y_one_hot, axis=0)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        Z = np.dot(X, self.weights) + self.bias
        return self._softmax(Z)

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)
