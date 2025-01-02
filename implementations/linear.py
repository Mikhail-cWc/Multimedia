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


class MyLinearRegression:
    def __init__(
        self,
        fit_intercept=True,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=None
    ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.coefficients_ = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def _mse_gradient(self, X, y, beta):
        y_pred = X.dot(beta)
        error = (y_pred - y)
        grad = (2 / X.shape[0]) * X.T.dot(error)
        return grad

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)  # (n_samples, 1)

        # Добавим столбец единиц, если fit_intercept=True
        X = self._add_intercept(X)

        n_samples, n_features = X.shape

        np.random.seed(12)
        beta = np.random.randn(n_features, 1)

        if self.batch_size is None or self.batch_size > n_samples:
            self.batch_size = n_samples

        for i in range(self.n_iterations):
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad = self._mse_gradient(X_batch, y_batch, beta)

                beta = beta - self.learning_rate * grad

        self.coefficients_ = beta

    def predict(self, X):
        X = np.array(X, dtype=float)
        X = self._add_intercept(X)  # учитываем intercept, если он есть

        y_pred = X.dot(self.coefficients_)

        return y_pred.ravel()
