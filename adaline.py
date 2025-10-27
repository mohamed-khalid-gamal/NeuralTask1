
import numpy as np


class Adaline:

    def __init__(self, learning_rate=0.01, n_epochs=100, mse_threshold=0.01,
                 use_bias=True, random_state=None):

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mse_threshold = mse_threshold
        self.use_bias = use_bias
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.mse_per_epoch = []

    def _initialize_weights(self, n_features):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = np.random.uniform(-0.5, 0.5, n_features)

        if self.use_bias:
            self.bias = np.random.uniform(-0.5, 0.5)
        else:
            self.bias = 0.0

    def _linear_activation(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):

        linear_output = self._linear_activation(X)
        return np.where(linear_output >= 0, 1, -1)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self._initialize_weights(n_features)

        for epoch in range(self.n_epochs):
            errors = []

            for i in range(n_samples):
                xi = X[i]
                ti = y[i]

                yi = np.dot(self.weights, xi) + self.bias

                error = ti - yi
                errors.append(error)

                self.weights += self.learning_rate * error * xi

                if self.use_bias:
                    self.bias += self.learning_rate * error

            errors = np.array(errors)
            mse = (1.0 / n_samples) * np.sum(0.5 * errors ** 2)
            self.mse_per_epoch.append(mse)

            if mse < self.mse_threshold:
                break

        return self

    def get_decision_boundary(self):

        if self.weights is None or len(self.weights) != 2:
            return None

        return self.weights[0], self.weights[1], self.bias

