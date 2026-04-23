import numpy as np
from typing import List

from beras.core import Diffable, Tensor

class Activation(Diffable):
    @property
    def weights(self) -> List[Tensor]: 
        return []

    def get_weight_gradients(self) -> List[Tensor]: 
        return []


class LeakyReLU(Activation):

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        out = np.where(x > 0, x, self.alpha * x)
        return Tensor(out)

    def get_input_gradients(self) -> List[Tensor]:
        x = self.inputs[0]
        # Gradient: 1 for x>0, alpha for x<=0 (0 at exactly x=0)
        grad = np.where(x > 0, 1.0, np.where(x < 0, self.alpha, 0.0))
        return [Tensor(grad)]

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    # ReLU is LeakyReLU with alpha=0
    def __init__(self) -> None:
        super().__init__(alpha=0)


class Sigmoid(Activation):

    def forward(self, x: Tensor) -> Tensor:
        out = 1 / (1 + np.exp(-x))
        return Tensor(out)

    def get_input_gradients(self) -> List[Tensor]:
        y = self.outputs[0]
        grad = y * (1 - y)
        return [Tensor(grad)]

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J


class Softmax(Activation):

    def forward(self, x: Tensor) -> Tensor:
        # Subtract max for numerical stability before computing softmax
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        out = exps / np.sum(exps, axis=-1, keepdims=True)
        return Tensor(out)

    def get_input_gradients(self) -> List[Tensor]:
        x = self.inputs[0]
        y = self.outputs[0]
        batch_size, n = x.shape

        # Full Jacobian: diag(y) - outer(y, y) per sample
        grad = np.zeros((batch_size, n, n), dtype=x.dtype)
        for b in range(batch_size):
            grad[b] = np.diag(y[b]) - np.outer(y[b], y[b])

        return [Tensor(grad)]