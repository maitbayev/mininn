from typing import Optional

import numpy as np

from mininn.module import Module


class MaxPool2D(Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: int = 0
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self._padded_input = np.ndarray([])

    def forward(self, input: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        if self.padding > 0:
            p = (self.padding, self.padding)
            input = np.pad(input, ((0, 0), (0, 0), p, p), constant_values=-np.inf)
        self._padded_input = input
        # (B, C, H, W)
        batches, c, h, w = input.shape
        h_out, w_out = (h - k) // self.stride + 1, (w - k) // self.stride + 1
        output = np.zeros([batches, c, h_out, w_out])
        for i in range(0, h - k + 1, self.stride):
            for j in range(0, w - k + 1, self.stride):
                window = input[:, :, i : i + k, j : j + k].max(axis=(2, 3))
                output[:, :, i // self.stride, j // self.stride] = window
        return output

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        batches, c, h, w = self._padded_input.shape
        grad_input = np.zeros_like(self._padded_input)
        for i in range(0, h - k + 1, self.stride):
            for j in range(0, w - k + 1, self.stride):
                window = self._padded_input[:, :, i : i + k, j : j + k].reshape(
                    (batches, c, -1)
                )
                index = np.argmax(window, axis=2)
                for b in range(batches):
                    for ci in range(c):
                        grad_input[
                            b, ci, index[b][ci] // k + i, index[b][ci] % k + j
                        ] += gradients[b, ci, i // self.stride, j // self.stride]
        if self.padding > 0:
            p = self.padding
            grad_input = grad_input[:, :, p:-p, p:-p]
        return grad_input
