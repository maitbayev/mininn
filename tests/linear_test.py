import typing
from typing import Optional

import numpy as np
import torch
from pytest import approx

from mininn import Linear


class ResultGradients:
    def __init__(self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        self.input = input
        self.weight = weight
        self.bias = bias


def _torch_linear(
    input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]
) -> np.ndarray:
    return torch.nn.functional.linear(
        torch.tensor(input),
        weight=torch.tensor(weight.T),
        bias=None if bias is None else torch.tensor(bias),
    ).numpy(force=True)


@typing.no_type_check
def _torch_linear_grad(
    input: np.ndarray, weight: np.ndarray, bias: np.ndarray, grad: np.ndarray
) -> ResultGradients:
    input = torch.tensor(input, requires_grad=True)  # type: ignore
    weight = torch.tensor(weight.T, requires_grad=True)  # type: ignore
    bias = torch.tensor(bias, requires_grad=True)  # type: ignore
    grad = torch.tensor(grad)  # type: ignore
    output = torch.nn.functional.linear(input, weight, bias)
    output.backward(grad)
    return ResultGradients(
        input.grad.numpy(force=True),
        weight.grad.numpy(force=True).T,
        bias.grad.numpy(force=True),
    )


def test_forward():
    for i in range(10):
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(10, in_features)
        assert linear.forward(input) == approx(
            _torch_linear(input, linear.weight.value, linear.bias.value)  # type: ignore
        )


def test_backward_input():
    for i in range(10):
        batches = 10
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(batches, in_features)
        grad = np.random.randn(batches, out_features)
        linear.forward(input)
        expected = _torch_linear_grad(
            input,
            linear.weight.value,
            linear.bias.value,  # type: ignore
            grad,
        )
        assert linear.backward(grad) == approx(expected.input)
        assert linear.weight.grad == approx(expected.weight)
        assert linear.bias.grad == approx(expected.bias)  # type: ignore


def test_parameters():
    l1 = Linear(in_features=3, out_features=5)
    assert list(l1.parameters()) == [l1.weight, l1.bias]
    l2 = Linear(in_features=3, out_features=5, bias=False)
    assert list(l2.parameters()) == [l2.weight]
