from typing import Iterable

import numpy as np

from mininn.module import Module
from mininn.parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.input = np.array([])
        self.in_features = in_features
        self.out_features = out_features
        lim = 1 / np.sqrt(in_features)
        self.weight = Parameter.uniform((in_features, out_features), lim)
        self.bias = Parameter.uniform(out_features, lim) if bias else None

    def forward(self, input: np.ndarray) -> np.ndarray:
        # (B, N) x (N, M) + (B,)
        self.input = input
        out = input @ self.weight.value
        if self.bias is not None:
            out += self.bias.value
        return out

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        self.weight.accumulate_grad(self.input.T @ gradients)
        if self.bias:
            self.bias.accumulate_grad(gradients.sum(axis=0))
        return gradients @ self.weight.value.T

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:
        yield self.weight
        if self.bias:
            yield self.bias
