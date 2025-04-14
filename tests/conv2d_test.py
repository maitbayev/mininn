import typing

import numpy as np
import torch
from pytest import approx

import mininn


class ResultGradients:
    def __init__(self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        self.input = input
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_torch(cls, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        return cls(
            input.numpy(force=True), weight.numpy(force=True), bias.numpy(force=True)
        )


def _torch_conv(
    conv: mininn.Conv2d,
    torch_conv: torch.nn.Conv2d,
    input: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(conv.weight.value))
        if conv.bias and torch_conv.bias is not None:
            torch_conv.bias.copy_(torch.tensor(conv.bias.value))
    input_tensor = torch.tensor(input)
    return torch_conv(input_tensor).numpy(force=True)


@typing.no_type_check
def _torch_conv_backward(
    conv: mininn.Conv2d,
    torch_conv: torch.nn.Conv2d,
    input: np.ndarray,
    grad: np.ndarray,
) -> ResultGradients:
    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(conv.weight.value))
        if conv.bias and torch_conv.bias is not None:
            torch_conv.bias.copy_(torch.tensor(conv.bias.value))
    input_tensor = torch.tensor(input, requires_grad=True)
    grad_tensor = torch.tensor(grad)
    output_tensor = torch_conv(input_tensor)
    output_tensor.backward(grad_tensor)
    return ResultGradients.from_torch(
        input_tensor.grad, torch_conv.weight.grad, torch_conv.bias.grad
    )


def _check_forward(conv, torch_conv):
    for _ in range(10):
        inp = np.random.randn(3, conv.in_channels, 6, 8)
        assert conv(inp) == approx(_torch_conv(conv, torch_conv, inp), abs=1e-6)


def _check_backward(conv, torch_conv):
    for _ in range(10):
        inp = np.random.randn(3, conv.in_channels, 6, 8)
        out = conv(inp)
        grad = np.random.randn(*out.shape)
        torch_grads = _torch_conv_backward(conv, torch_conv, inp, grad)
        assert conv.backward(grad) == approx(torch_grads.input, abs=1e-6)
        assert conv.weight.grad == approx(torch_grads.weight, abs=1e-6)
        assert conv.bias.grad == approx(torch_grads.bias, abs=1e-6)


def test_forward():
    torch.set_default_dtype(torch.float64)
    conv = mininn.Conv2d(4, 5, 3)
    torch_conv = torch.nn.Conv2d(4, 5, 3)
    _check_forward(conv, torch_conv)


def test_forward_padding_same():
    torch.set_default_dtype(torch.float64)
    conv = mininn.Conv2d(4, 5, 3, padding="same")
    torch_conv = torch.nn.Conv2d(4, 5, 3, padding="same")
    _check_forward(conv, torch_conv)


def test_backward():
    torch.set_default_dtype(torch.float64)
    conv = mininn.Conv2d(4, 5, 3)
    torch_conv = torch.nn.Conv2d(4, 5, 3)
    _check_backward(conv, torch_conv)


def test_backward_padding_same():
    torch.set_default_dtype(torch.float64)
    conv = mininn.Conv2d(4, 5, 3, padding="same")
    torch_conv = torch.nn.Conv2d(4, 5, 3, padding="same")
    _check_backward(conv, torch_conv)
