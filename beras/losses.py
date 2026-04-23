import numpy as np
from typing import List

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> List[Tensor]:
        return []

    def get_weight_gradients(self) -> List[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # squared differences
        squared_error = (y_true - y_pred) ** 2
        # mean over features, then mean over batch
        loss = np.mean(np.mean(squared_error, axis=1))
        return Tensor(loss)

    def get_input_gradients(self) -> List[Tensor]:
        y_pred, y_true = self.inputs[0], self.inputs[1]
        batch_size, num_features = y_true.shape
        grad_pred = 2 * (y_pred - y_true) / (batch_size * num_features)
        grad_true = np.zeros_like(y_true)
        return [Tensor(grad_pred), Tensor(grad_true)]


class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Categorical cross entropy forward pass!"""
        # clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        # per-sample loss
        per_sample_loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        # mean over batch
        loss = np.mean(per_sample_loss)
        return Tensor(loss)

    def get_input_gradients(self) -> List[Tensor]:
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs[0], self.inputs[1]
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        grad_pred = -(y_true / y_pred_clipped) / y_true.shape[0]
        grad_true = np.zeros_like(y_true)
        return [Tensor(grad_pred), Tensor(grad_true)]
