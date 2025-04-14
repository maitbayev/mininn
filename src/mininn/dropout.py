import numpy as np

from mininn.module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = np.ndarray([])
        self.input = np.ndarray([])

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.is_training():
            self.input = input
            self.mask = np.random.binomial(1, 1 - self.p, size=self.input.shape)
            return self.mask * input * 1.0 / (1.0 - self.p)
        else:
            return input

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        if self.is_training():
            return gradients * self.mask * 1.0 / (1.0 - self.p)
        else:
            return gradients
