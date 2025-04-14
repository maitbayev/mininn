import numpy as np
import torch
from pytest import approx

import mininn
import mininn.optim


def _make_torch_model(
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray
) -> torch.nn.Sequential:
    linear1 = torch.nn.Linear(
        in_features=w1.shape[0],
        out_features=w1.shape[1],
        bias=True,
        dtype=torch.float64,
    )
    linear2 = torch.nn.Linear(
        in_features=w2.shape[0],
        out_features=w2.shape[1],
        bias=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        linear1.weight.copy_(torch.from_numpy(w1.T))
        linear1.bias.copy_(torch.from_numpy(b1))
        linear2.weight.copy_(torch.from_numpy(w2.T))
    return torch.nn.Sequential(
        linear1,
        torch.nn.ReLU(),
        linear2,
        torch.nn.ReLU(),
    )


def _make_model(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray) -> mininn.Sequential:
    linear1 = mininn.Linear(in_features=w1.shape[0], out_features=w1.shape[1])
    linear2 = mininn.Linear(
        in_features=w2.shape[0], out_features=w2.shape[1], bias=False
    )
    linear1.weight.set(w1)
    linear1.bias.set(b1)  # type: ignore
    linear2.weight.set(w2)
    return mininn.Sequential(
        [
            linear1,
            mininn.ReLU(),
            linear2,
            mininn.ReLU(),
        ]
    )


def check_optimizer(
    model: mininn.Sequential,
    optimizer: mininn.optim.Optimizer,
    torch_model: torch.nn.Sequential,
    torch_optimizer: torch.optim.Optimizer,
):
    loss = mininn.MSELoss()
    for i in range(10):
        x = np.random.randn(10, 3)
        target = np.random.randn(10, 7)
        x_t = torch.from_numpy(x.copy())
        target_t = torch.from_numpy(target.copy())

        torch_optimizer.zero_grad()
        torch_loss = torch.nn.functional.mse_loss(torch_model(x_t), target_t)
        torch_loss.backward()
        torch_optimizer.step()

        optimizer.zero_grad()
        loss(model(x), target)
        model.backward(loss.backward())
        optimizer.step()

        for module, torch_module in zip(model.modules, torch_model.children()):
            params = list(module.parameters(recurse=False))
            torch_params = list(torch_module.named_parameters(recurse=False))
            assert len(params) == len(torch_params)
            for p, (t_name, t) in zip(params, torch_params):
                w = t.numpy(force=True)
                if "weight" in t_name:
                    w = w.T
                assert p.value == approx(w), t_name


def test_sgd():
    torch.set_default_dtype(torch.float64)
    for _ in range(10):
        w1 = np.random.randn(3, 5)
        b1 = np.random.randn(5)
        w2 = np.random.randn(5, 7)
        torch_model = _make_torch_model(w1, b1, w2)
        model = _make_model(w1, b1, w2)
        torch_sgd = torch.optim.SGD(torch_model.parameters(), lr=0.1)
        sgd = mininn.optim.SGD(model.parameters(), lr=0.1)
        check_optimizer(model, sgd, torch_model, torch_sgd)


def test_sgd_momentum():
    torch.set_default_dtype(torch.float64)
    for _ in range(10):
        w1 = np.random.randn(3, 5)
        b1 = np.random.randn(5)
        w2 = np.random.randn(5, 7)
        torch_model = _make_torch_model(w1, b1, w2)
        model = _make_model(w1, b1, w2)
        torch_sgd = torch.optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.9)
        sgd = mininn.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        check_optimizer(model, sgd, torch_model, torch_sgd)


def test_rmsprop():
    torch.set_default_dtype(torch.float64)
    for _ in range(10):
        w1 = np.random.randn(3, 5)
        b1 = np.random.randn(5)
        w2 = np.random.randn(5, 7)
        torch_model = _make_torch_model(w1, b1, w2)
        model = _make_model(w1, b1, w2)
        torch_sgd = torch.optim.RMSprop(torch_model.parameters(), lr=0.1)
        sgd = mininn.optim.RMSprop(model.parameters(), lr=0.1)
        check_optimizer(model, sgd, torch_model, torch_sgd)


def test_adam():
    torch.set_default_dtype(torch.float64)
    for _ in range(10):
        w1 = np.random.randn(3, 5)
        b1 = np.random.randn(5)
        w2 = np.random.randn(5, 7)
        torch_model = _make_torch_model(w1, b1, w2)
        model = _make_model(w1, b1, w2)
        torch_sgd = torch.optim.Adam(torch_model.parameters(), lr=0.1)
        sgd = mininn.optim.Adam(model.parameters(), lr=0.1)
        check_optimizer(model, sgd, torch_model, torch_sgd)
