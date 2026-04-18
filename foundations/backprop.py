import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        L = 0.5 * (y_hat - y_true)**2
        dL_dw = []
        cm = (y_hat - y_true) * y_hat * (1-y_hat)
        dbi = cm
        for j in range(w.shape[0]):
            dwi = cm * x[j]
            dL_dw.append(dwi)
        dL_dw = np.round(np.array(dL_dw), 5)
        dL_db = np.round(dbi, 5)

        return dL_dw, dL_db
            
