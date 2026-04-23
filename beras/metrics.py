import numpy as np

from beras.core import Callable, Tensor


class CategoricalAccuracy(Callable):
    def forward(self, probs: Tensor, labels: Tensor) -> float:
        pred_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(labels, axis=1)
        acc = np.mean(pred_classes == true_classes)
        return Tensor(acc)
