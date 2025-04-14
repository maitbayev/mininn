import numpy as np
import torch
from pytest import approx

from mininn import Softmax


def _torch_softmax(input: np.ndarray) -> np.ndarray:
    return torch.nn.functional.softmax(torch.Tensor(input), dim=1).numpy(force=True)


def _torch_softmax_backward(input: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    input_tensor = torch.tensor(input, requires_grad=True)
    output = torch.nn.functional.softmax(input_tensor, dim=1)
    output.backward(torch.Tensor(gradient).requires_grad_(True))
    return input_tensor.grad.numpy(force=True)  # type: ignore


def test_forward():
    softmax = Softmax()
    for i in range(10):
        inp = np.random.randn(3, 10)
        if i % 2 == 0:
            inp = inp.T
        assert softmax.forward(inp) == approx(_torch_softmax(inp))


def test_backward():
    softmax = Softmax()
    for i in range(10):
        inp = np.random.randn(3, 10)
        if i % 2 == 0:
            inp = inp.T
        gradient = np.random.randn(*inp.shape)
        softmax.forward(inp)
        g = softmax.backward(gradients=gradient)
        assert g == approx(
            _torch_softmax_backward(inp, gradient=gradient), rel=0.0001, abs=1e-6
        )
