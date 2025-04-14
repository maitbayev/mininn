import numpy as np
import torch
from pytest import approx

import mininn


def test_forward():
    inp = np.random.randn(2, 3, 4, 5)
    flatten = mininn.Flatten()
    assert flatten.forward(inp) == approx(
        torch.flatten(torch.tensor(inp), start_dim=1).numpy(force=True)
    )


def test_forward1():
    inp = np.random.randn(2, 3, 4, 5, 6)
    flatten = mininn.Flatten(start_dim=2, end_dim=-2)
    assert flatten.forward(inp) == approx(
        torch.flatten(torch.tensor(inp), start_dim=2, end_dim=-2).numpy(force=True)
    )


def test_backward():
    inp = np.random.randn(2, 3, 4, 5, 6)
    grad = np.random.randn(2, 3 * 4 * 5 * 6)
    inp_tensor = torch.tensor(inp, requires_grad=True)
    out_tensor = torch.flatten(inp_tensor, start_dim=1)
    out_tensor.backward(torch.tensor(grad))

    flatten = mininn.Flatten()
    flatten.forward(inp)

    assert inp_tensor.grad is not None
    assert flatten.backward(grad) == approx(inp_tensor.grad.numpy(force=True))


def test_backward1():
    inp = np.random.randn(2, 3, 4, 5, 6)
    grad = np.random.randn(2, 3, 4 * 5, 6)
    inp_tensor = torch.tensor(inp, requires_grad=True)
    out_tensor = torch.flatten(inp_tensor, start_dim=2, end_dim=-2)
    out_tensor.backward(torch.tensor(grad))

    flatten = mininn.Flatten(start_dim=2, end_dim=-2)
    flatten.forward(inp)

    assert inp_tensor.grad is not None
    assert flatten.backward(grad) == approx(inp_tensor.grad.numpy(force=True))
