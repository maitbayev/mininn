import typing

import numpy as np
import torch
from pytest import approx

import mininn


def _torch_sequence(w1: np.ndarray, b1: np.ndarray) -> torch.nn.Sequential:
    linear = torch.nn.Linear(
        in_features=w1.shape[0], out_features=w1.shape[1], dtype=torch.float64
    )
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(w1.T))
        linear.bias.copy_(torch.tensor(b1))
    return torch.nn.Sequential(
        linear,
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=1),
    )


@typing.no_type_check
def _torch_backward(
    seq: torch.nn.Sequential, input: np.ndarray, grad: np.ndarray
) -> np.ndarray:
    input = torch.tensor(input, requires_grad=True)
    output = seq(input)
    output.backward(torch.tensor(grad))
    return input.grad.numpy(force=True)


def _make_sequence(w1: np.ndarray, b1: np.ndarray) -> mininn.Sequential:
    linear = mininn.Linear(in_features=w1.shape[0], out_features=w1.shape[1])
    linear.weight.value = w1
    if linear.bias:
        linear.bias.value = b1
    return mininn.Sequential(
        [
            linear,
            mininn.ReLU(),
            mininn.Softmax(),
        ]
    )


def test_forward():
    for _ in range(10):
        x = np.random.randn(3, 5)
        w1 = np.random.randn(5, 7)
        b1 = np.random.randn(7)
        seq = _make_sequence(w1, b1)
        torch_seq = _torch_sequence(w1, b1)
        assert seq(x) == approx(torch_seq(torch.tensor(x)).numpy(force=True))


def test_backward():
    for _ in range(10):
        x = np.random.randn(3, 5)
        w1 = np.random.randn(5, 7)
        b1 = np.random.randn(7)
        seq = _make_sequence(w1, b1)
        grad = np.random.randn(3, 7)
        torch_seq = _torch_sequence(w1, b1)
        torch_grad = _torch_backward(torch_seq, x, grad)
        seq.forward(x)
        assert seq.backward(grad) == approx(torch_grad)


def test_parameters():
    s0 = mininn.Sequential(
        [
            mininn.ReLU(),
        ]
    )
    assert list(s0.parameters()) == []
    l1 = mininn.Linear(in_features=5, out_features=5)
    l2 = mininn.Linear(in_features=5, out_features=8, bias=False)
    s1 = mininn.Sequential(
        [
            l1,
            mininn.ReLU(),
            mininn.Softmax(),
        ]
    )
    assert list(s1.parameters()) == [l1.weight, l1.bias]
    s2 = mininn.Sequential([l1, l2])
    assert list(s2.parameters()) == [l1.weight, l1.bias, l2.weight]
