import numpy as np

from mininn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = np.ndarray([])

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return (self.input >= 0).astype(gradients.dtype) * gradients
