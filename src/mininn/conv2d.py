from typing import Iterable, Literal

import numpy as np

from mininn.module import Module
from mininn.parameter import Parameter


class Conv2d(Module):
    PaddingLiteral = Literal["same", "valid"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLiteral = "valid",
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        lim = 1.0 / (in_channels * kernel_size * kernel_size)
        self.weight = Parameter.uniform(
            (out_channels, in_channels, kernel_size, kernel_size), range=lim
        )
        self.bias = Parameter.uniform(out_channels, range=lim) if bias else None
        self.padded_input = np.ndarray([])

    def forward(self, input: np.ndarray) -> np.ndarray:
        # (B, C, H, W) => (B, C', H', W')
        k = self.kernel_size
        if self.padding == "same":
            p = (k // 2, k // 2)
            input = np.pad(input, [(0, 0), (0, 0), p, p])
        self.padded_input = input

        batches, _, h, w = input.shape
        output = np.zeros([batches, self.out_channels, h - k + 1, w - k + 1])
        for i in range(h - k + 1):
            for j in range(w - k + 1):
                output[:, :, i, j] = np.tensordot(
                    input[:, :, i : i + k, j : j + k],
                    self.weight.value,
                    axes=((1, 2, 3), (1, 2, 3)),
                )
                if self.bias:
                    output[:, :, i, j] += self.bias.value
        return output

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        batches, _, h, w = self.padded_input.shape
        grad_input = np.zeros_like(self.padded_input)
        for i in range(h - k + 1):
            for j in range(w - k + 1):
                grad_input[:, :, i : i + k, j : j + k] += np.einsum(
                    "dcij,bd->bcij", self.weight.value, gradients[:, :, i, j]
                )
                self.weight.accumulate_grad(
                    np.einsum(
                        "bcij,bd->dcij",
                        self.padded_input[:, :, i : i + k, j : j + k],
                        gradients[:, :, i, j],
                    )
                )
        if self.bias:
            self.bias.accumulate_grad(gradients.sum(axis=(0, 2, 3)))
        if self.padding == "same":
            p = k // 2
            grad_input = grad_input[:, :, p:-p, p:-p]
        return grad_input

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:
        yield self.weight
        if self.bias:
            yield self.bias
