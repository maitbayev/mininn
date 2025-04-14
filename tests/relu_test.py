import typing

import numpy as np
import torch
from pytest import approx

from mininn import ReLU


def _torch_relu(input: np.ndarray) -> np.ndarray:
    return torch.nn.functional.relu(torch.tensor(input)).numpy(force=True)


@typing.no_type_check
def _torch_relu_grad(input: np.ndarray, grad: np.ndarray) -> np.ndarray:
    input = torch.tensor(input, requires_grad=True)
    output = torch.nn.functional.relu(input)
    output.backward(torch.tensor(grad))
    return input.grad.numpy(force=True)


def test_forward():
    relu = ReLU()
    for i in range(10):
        input = np.random.randn(3, 5) * 10
        if i % 2 == 0:
            input = input.T
        assert relu(input) == approx(_torch_relu(input))


def test_backward():
    relu = ReLU()
    for i in range(10):
        input = np.random.randn(3, 5) * 10
        if i % 2 == 0:
            input = input.T
        grad = np.random.randn(*input.shape)
        relu(input)
        assert relu.backward(grad) == approx(_torch_relu_grad(input, grad))
