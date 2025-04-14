import math

import numpy as np

from mininn.module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.input_shape = ()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input_shape = input.shape
        return input.reshape(
            _flattened_shape(input.shape, self.start_dim, self.end_dim)
        )

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return gradients.reshape(self.input_shape)


def _flattened_shape(
    shape: tuple[int, ...], start_dim: int, end_dim: int
) -> tuple[int, ...]:
    start = start_dim
    stop = len(shape) if end_dim == -1 else end_dim + 1
    return shape[:start] + (math.prod(shape[start:stop]),) + shape[stop:]
