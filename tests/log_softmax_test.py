import numpy as np
import torch
from pytest import approx

from mininn import LogSoftmax


def _torch_log_softmax(input: np.ndarray) -> np.ndarray:
    return torch.nn.functional.log_softmax(torch.tensor(input), dim=1).numpy(force=True)


def _torch_log_softmax_backward(input: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    input_tensor = torch.tensor(input, requires_grad=True)
    output = torch.nn.functional.log_softmax(input_tensor, dim=1)
    output.backward(torch.tensor(gradient))
    return input_tensor.grad.numpy(force=True)  # type: ignore


def test_forward():
    log_softmax = LogSoftmax()
    for i in range(10):
        inp = np.random.randn(3, 10)
        if i % 2 == 0:
            inp = inp.T
        assert log_softmax.forward(inp) == approx(_torch_log_softmax(inp))


def test_backward():
    log_softmax = LogSoftmax()
    for i in range(10):
        inp = np.random.randn(3, 10)
        if i % 2 == 0:
            inp = inp.T
        gradient = np.random.randn(*inp.shape)
        log_softmax.forward(inp)
        g = log_softmax.backward(gradients=gradient)
        assert g == approx(_torch_log_softmax_backward(inp, gradient=gradient))
