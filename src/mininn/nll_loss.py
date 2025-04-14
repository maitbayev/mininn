import numpy as np

from mininn.module import Module


class NLLLoss(Module):
    def __init__(self):
        super().__init__()
        self.target = np.array([])
        self.input_shape = np.array([])

    def forward(self, input, target):
        batches = input.shape[0]
        # input (B, N)
        # target (N,)
        sum = -input[np.arange(batches), target].sum()
        self.target = target
        self.input_shape = input.shape
        return sum / batches

    def backward(self, gradients=1) -> np.ndarray:
        batches = self.input_shape[0]
        output = np.zeros(self.input_shape)
        output[np.arange(batches), self.target] = -1.0 / batches
        return output
