import random

import numpy as np
import torch
from pytest import approx

import mininn


def test_forward():
    for _ in range(10):
        k = random.randint(1, 3) * 2 + 1
        stride = random.randint(2, 5)
        padding = random.randint(0, k // 2)
        inp = np.random.rand(4, 5, np.random.randint(10, 20), np.random.randint(10, 20))
        max_pool2d = mininn.MaxPool2D(k, stride, padding)
        inp_torch = torch.tensor(inp)
        max_pool2d_torch = torch.nn.functional.max_pool2d(
            inp_torch, k, stride, padding=padding
        )
        assert max_pool2d(inp) == approx(max_pool2d_torch.numpy(force=True))


def test_backward():
    for _ in range(100):
        k = random.randint(1, 3) * 2 + 1
        stride = random.randint(2, 5)
        padding = random.randint(0, k // 2)
        inp = np.random.rand(4, 5, 10, 10)
        max_pool2d = mininn.MaxPool2D(k, stride, padding)
        out = max_pool2d(inp)
        grad: np.ndarray = np.random.rand(*out.shape)  # type: ignore

        inp_torch = torch.tensor(inp, requires_grad=True)
        max_pool2d_torch = torch.nn.functional.max_pool2d(
            inp_torch, k, stride, padding=padding
        )
        max_pool2d_torch.backward(torch.tensor(grad))

        assert inp_torch.grad is not None
        assert max_pool2d.backward(grad) == approx(inp_torch.grad.numpy(force=True))
