import numpy as np

from mininn.log_softmax import LogSoftmax
from mininn.module import Module
from mininn.nll_loss import NLLLoss


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.nll_loss = NLLLoss()

    def forward(self, input, target):
        return self.nll_loss.forward(self.log_softmax(input), target)

    def backward(self) -> np.ndarray:
        g = self.nll_loss.backward()
        return self.log_softmax.backward(g)
