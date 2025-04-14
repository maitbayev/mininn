import numpy as np
from numpy.typing import NDArray

from mininn.module import Module


class Softmax(Module):
    def __init__(self):
        self.softmax = np.array([])

    def forward(self, input: NDArray) -> NDArray:
        mx = np.max(input, axis=1, keepdims=True)
        e = np.exp(input - mx)
        self.softmax = e / np.sum(e, axis=1, keepdims=True)
        return self.softmax

    def backward(self, gradients: NDArray) -> NDArray:
        batches = self.softmax.shape[0]
        results = []
        for b in range(batches):
            s = self.softmax[b : b + 1]
            j = np.diagflat(s) - s.T @ s
            results.append(gradients[b] @ j)
        return np.array(results)
