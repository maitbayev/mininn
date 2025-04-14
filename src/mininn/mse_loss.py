import numpy as np

from mininn.module import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.input = np.array([])
        self.target = np.array([])

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input = input
        self.target = target
        return np.square(input - target).mean()

    def backward(self) -> np.ndarray:
        return 2.0 * (self.input - self.target) / self.input.size
