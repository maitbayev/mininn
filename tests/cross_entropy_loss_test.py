import numpy as np
import torch
from pytest import approx

from mininn import CrossEntropyLoss


def _torch_cross_entropy(predicted, target):
    return torch.nn.functional.cross_entropy(
        torch.tensor(predicted), torch.tensor(target)
    )


def _torch_cross_entropy_grad(predicted, target):
    predicted = torch.tensor(predicted, requires_grad=True)
    target = torch.tensor(target)
    output = torch.nn.functional.cross_entropy(predicted, target)
    output.backward()
    return predicted.grad


def test_forward():
    loss = CrossEntropyLoss()
    for _ in range(10):
        predicted = np.random.randn(5, 10) * 10
        target = np.random.randint(low=0, high=10, size=predicted.shape[0])
        assert loss.forward(predicted, target) == approx(
            _torch_cross_entropy(predicted, target)
        )


def test_backward():
    loss = CrossEntropyLoss()
    for _ in range(10):
        predicted = np.random.randn(5, 10) * 10
        target = np.random.randint(low=0, high=10, size=predicted.shape[0])
        loss.forward(predicted, target)
        assert loss.backward() == approx(_torch_cross_entropy_grad(predicted, target))
