import numpy as np
from numpy.typing import NDArray

from mininn.module import Module


class LogSoftmax(Module):
    def __init__(self):
        self.e = np.array([])
        self.log_softmax = np.array([])

    def forward(self, input: NDArray) -> NDArray:
        # log(e[i]/sum e[j])
        #   = log(e^x[i]) - log(sum e^x[j])
        #   = x[i] - log(sum e^x[j])
        #   = x[i] - log(e^mx * sum  e^(x[j]-mx))
        #   = x[i] - mx - log(sum e^(x[j]-max))

        # (B, 1)
        shifted = input - np.max(input, axis=1, keepdims=True)
        # (B, N)
        self.e = np.exp(shifted)
        self.log_softmax = shifted - np.log(np.sum(self.e, axis=1, keepdims=True))
        return self.log_softmax

    def backward(self, gradients: NDArray) -> NDArray:
        # if i != j:
        #   d log(e[i]/sum )  / d x[j]
        #       = - sum / e[i] * (e[i] / sum) * (e[j] / sum)
        #       = - e[j] / sum
        # if i == j:
        #   d log(e[i]/sum ) / d x[j]
        #       = sum / e[i] * (e[i]/sum) * (1 - e[i]/sum)
        #       = 1 - e[i] / sum
        batches, n = self.log_softmax.shape
        eye = np.eye(n).reshape(1, n, n)
        # (batches, n) -> (batches, 1, n)
        e = np.expand_dims(self.e, axis=1)
        # (batches, n, n) = (1, n, n) - (batches, 1, n) / (batches, 1, 1)
        j = eye - e / e.sum(axis=2, keepdims=True)

        # einsum: bj,bji->bi
        # (batches, n) -> (batches, 1, n)
        gradients = np.expand_dims(gradients, axis=1)
        # (batches, 1, n) @ (batches, n, n) -> (batches, n)
        return np.squeeze(gradients @ j, axis=1)
