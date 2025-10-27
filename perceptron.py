
import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_epochs=100, use_bias=True, random_state=None):

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def _initialize_weights(self, n_features):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Small random weights
        self.weights = np.random.uniform(-0.5, 0.5, n_features)

        if self.use_bias:
            self.bias = np.random.uniform(-0.5, 0.5)
        else:
            self.bias = 0.0

    def _activation(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self._initialize_weights(n_features)

        for epoch in range(self.n_epochs):
            errors = 0

            for i in range(n_samples):
                xi = X[i]
                yi_true = y[i]


                linear_output = np.dot(xi, self.weights) + self.bias
                yi_pred = self._activation(linear_output)

                if yi_pred != yi_true:
                    error = yi_true - yi_pred

                    self.weights += self.learning_rate * error * xi

                    if self.use_bias:
                        self.bias += self.learning_rate * error

                    errors += 1

            self.errors_per_epoch.append(errors)

            if errors == 0:
                break

        return self

    def get_decision_boundary(self):

        if self.weights is None or len(self.weights) != 2:
            return None

        return self.weights[0], self.weights[1], self.bias

