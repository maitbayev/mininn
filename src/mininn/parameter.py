from typing import Optional, Self, final

import numpy as np


@final
class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad: Optional[np.ndarray] = None

    @classmethod
    def uniform(cls, shape: tuple | int, range: float) -> Self:
        return cls(np.random.uniform(low=-range, high=range, size=shape))

    def set(self, value: np.ndarray) -> Self:
        self.value = value
        return self

    def add(self, delta: np.ndarray, alpha: float = 1) -> Self:
        self.value += alpha * delta
        return self

    def accumulate_grad(self, delta: np.ndarray):
        if self.grad is None:
            self.grad = delta
        else:
            self.grad += delta
