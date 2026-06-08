import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n = X.shape[0]
        f = X.shape[1]

        w = np.zeros(f)
        b = 0

        for epoch in range(epochs):
            y_hat = X @ w + b
            MSE = np.mean((y_hat - y)**2)
            gr_w = 2/n * X.T @ (y_hat - y)
            gr_b = 2 * np.mean(y_hat - y)
            
            w = w - lr * gr_w
            b = b - lr * gr_b

        return np.round(w, 5), np.round(b, 5)
