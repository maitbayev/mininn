import numpy as np
import torch
from pytest import approx

from mininn import NLLLoss


def _torch_nll_loss(predicted, target):
    return torch.nn.functional.nll_loss(torch.tensor(predicted), torch.tensor(target))


def _torch_nll_loss_grad(predicted, target):
    predicted = torch.tensor(predicted, requires_grad=True)
    target = torch.tensor(target)
    output = torch.nn.functional.nll_loss(predicted, target)
    output.backward()
    return predicted.grad


def test_forward():
    nll = NLLLoss()
    for _ in range(10):
        predicted = np.random.randn(5, 10) * 10
        target = np.random.randint(low=0, high=10, size=predicted.shape[0])
        assert nll.forward(predicted, target) == approx(
            _torch_nll_loss(predicted, target)
        )


def test_backward():
    nll = NLLLoss()
    for _ in range(10):
        predicted = np.random.randn(5, 10) * 10
        target = np.random.randint(low=0, high=10, size=predicted.shape[0])
        nll.forward(predicted, target)
        assert nll.backward() == approx(_torch_nll_loss_grad(predicted, target))
