import numpy as np
from pytest import approx

from mininn import Dropout


def test_smoke_training():
    dropout = Dropout(p=0.4)
    out = dropout(np.ones((5, 20)))
    dropout.backward(out)


def test_eval():
    x = np.random.randn(3, 5, 6)
    dropout = Dropout(p=0.4)
    dropout.set_training(False)
    assert dropout(x) == approx(x)
    assert dropout.backward(x) == approx(x)


def test_prob1():
    x = np.random.randn(3, 5, 6)
    dropout = Dropout(p=0.9999999)
    assert dropout(x) == approx(np.zeros_like(x))


def test_prob0():
    x = np.random.randn(3, 5, 6)
    dropout = Dropout(p=0)
    assert dropout.is_training()
    assert dropout(x) == approx(x)

    dropout.set_training(False)
    assert dropout(x) == approx(x)
