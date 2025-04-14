import typing

import numpy as np
import torch
from pytest import approx

from mininn import MSELoss


def _torch_mse_loss(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    return torch.nn.functional.mse_loss(
        torch.tensor(input), torch.tensor(target)
    ).numpy(force=True)


@typing.no_type_check
def _torch_mse_loss_grad(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    input_torch = torch.tensor(input, requires_grad=True)
    target = torch.tensor(target)
    output = torch.nn.functional.mse_loss(input_torch, target)
    output.backward()
    return input_torch.grad.numpy()


def test_forward():
    mse = MSELoss()
    for _ in range(10):
        input = np.random.randn(10, 3)
        target = np.random.randn(10, 3)
        _torch_mse_loss(input, target)
        assert mse(input, target) == approx(_torch_mse_loss(input, target))


def test_backward():
    mse = MSELoss()
    for _ in range(10):
        input = np.random.randn(3, 10)
        target = np.random.randn(3, 10)
        _torch_mse_loss(input, target)
        mse.forward(input, target)

        assert mse.backward() == approx(_torch_mse_loss_grad(input, target))
