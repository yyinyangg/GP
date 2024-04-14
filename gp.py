from kernels import Kernels
import numpy as np


class MyGP:
    def __init__(self):
        self.k = Kernels.get_rbf(sigma=1, lengthscale=1)
        self.mean = np.array([])
        self.alpha = np.array([])
        self.X_train = np.array([])
        self.Y_train = np.array([])

    def mean_function(self, X_train):
        self.mean = np.zeros(len(X_train))

    def sample(self, X_train, sample_num):
        self.mean_function(X_train)
        K = self.k(X_train, X_train)
        samples = []
        for _ in range(sample_num):
            sample = np.random.multivariate_normal(mean=self.mean, cov=K)
            samples.append(sample)
        return samples

    def fit(self, X_train, Y_train):
        assert len(X_train) == len(Y_train),"training data and label should have the same length"
        self.X_train, self.Y_train = X_train, Y_train
        self.mean_function(self.X_train)
        K = self.k(self.X_train)
        self.alpha = np.linalg.solve(K + 1e-8 * np.eye(len(self.X_train)), self.Y_train - self.mean)

    def predict(self, X_test):
        return np.dot(self.k(X_test, self.X_train), self.alpha)
