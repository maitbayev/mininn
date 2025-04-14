from typing import Iterable, Self

import numpy as np

from mininn.module import Module
from mininn.parameter import Parameter


class Sequential(Module):
    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        output = gradients
        for module in reversed(self.modules):
            output = module.backward(output)
        return output

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:
        if recurse:
            for module in self.modules:
                for param in module.parameters():
                    yield param

    def set_training(self, value: bool = True) -> Self:
        super().set_training(value)
        for module in self.modules:
            module.set_training(value)
        return self
