import numpy as np


class Kernels:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_rbf(sigma=1, lengthscale=1):
        return RBF(sigma, lengthscale).get_kernel


class KernelBase:
    def __init__(self):
        self.X = np.zeros(1)
        self.X_hat = np.zeros(self.X.shape)

    def check_data_type(self, x: np.ndarray, x_hat: np.ndarray) -> None:
        """
        Params:
        x: ndarray with the shape of (n,p), n is the number of data points, p is the dimension of data vector
        x_hat: ndarray with the shape of (n,p)
        """
        try:
            assert isinstance(x, np.ndarray) and isinstance(x_hat, np.ndarray), "data type must be ndarray"
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x_hat.ndim == 1:
                x_hat = x_hat.reshape(1, -1)
            assert x.shape[1] == x_hat.shape[1], "data must have same dimension"
            self.X = x
            self.X_hat = x_hat
        except AssertionError:
            print("data format error")
            exit(1)


class RBF(KernelBase):
    def __init__(self, variance=1, lengthscale=1) -> None:
        super().__init__()
        self.sigma = variance
        self.len = lengthscale

    def get_kernel(self, x: np.ndarray, x_hat: np.ndarray = None) -> np.ndarray:
        if x_hat is None:
            self.check_data_type(x, x)
            n= self.X.shape[0]
            K = np.zeros((n,n))
            for i in range(n):
                diff = np.sum((self.X[i] - self.X_hat[i:,:])**2, axis=1)
                result = self.sigma ** 2 * np.exp(-0.5 * diff / (self.len ** 2))
                K[i, i:] = result
                if i+1 < n:
                    K[i+1:, i] = result[i+1-n:]
        else:
            self.check_data_type(x, x_hat)
            n, m = self.X.shape[0], self.X_hat.shape[0]
            K = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    diff = np.sum((self.X[i,:] - self.X_hat[j,:])**2)
                    K[i,j] = self.sigma ** 2 * np.exp(-0.5 * diff / (self.len ** 2))
        return K
