from abc import abstractmethod
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Any, Optional

from beras.core import Diffable, Tensor, Variable, Callable
from beras.gradient_tape import GradientTape
import numpy as np

def print_stats(stat_dict: Dict[str, float], batch_num: Optional[int] = None, num_batches: Optional[int] = None, epoch: Optional[int] = None, avg: bool = False) -> None:
    """
    Given a dictionary of names statistics and batch/epoch info,
    print them in an appealing manner. If avg, display stat averages.

    :param stat_dict: dictionary of metrics to display
    :param batch_num: current batch number
    :param num_batches: total number of batches
    :param epoch: current epoch number
    :param avg: whether to display averages
    """
    title_str = " - "
    if epoch is not None:
        title_str += f"Epoch {epoch+1:2}: "
    if batch_num is not None:
        title_str += f"Batch {batch_num+1:3}"
        if num_batches is not None:
            title_str += f"/{num_batches}"
    if avg:
        title_str += f"Average Stats"
    print(f"\r{title_str} : ", end="")
    op = np.mean if avg else lambda x: x
    print({k: np.round(op(v), 4) for k, v in stat_dict.items()}, end="")
    print("   ", end="" if not avg else "\n")


def update_metric_dict(super_dict: Dict[str, List[float]], sub_dict: Dict[str, float]) -> None:
    """
    Appends the average of the sub_dict metrics to the super_dict's metric list

    :param super_dict: dictionary of metrics to append to
    :param sub_dict: dictionary of metrics to average and append
    """
    for k, v in sub_dict.items():
        super_dict[k] += [np.mean(v)]


class Model(Diffable):

    def __init__(self, layers: List[Diffable]) -> None:
        """
        Initialize all trainable parameters and take layers as inputs
        """
        # Initialize all trainable parameters
        self.layers = layers

    @property
    def weights(self) -> List[Tensor]:
        """
        Return the weights of the model by iterating through the layers
        """
        return [w for layer in self.layers for w in layer.weights]

    def compile(self, optimizer: Diffable, loss_fn: Diffable, acc_fn: Callable) -> None:
        """
        "Compile" the model by taking in the optimizers, loss, and accuracy functions.
        In more optimized DL implementations, this will have more involved processes
        that make the components extremely efficient but very inflexible.
        """
        self.optimizer      = optimizer
        self.compiled_loss  = loss_fn
        self.compiled_acc   = acc_fn

    def fit(self, x: Tensor, y: Union[Tensor, np.ndarray], epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Trains the model by iterating over the input dataset and feeding input batches
        into the batch_step method with training. At the end, the metrics are returned.

        Args:
            x: Tensor, the dataset inputs
            y: Tensor, the dataset labels
            epochs: int, the number of epochs
            batch_size: int, the batch size

        Returns:
            Dict[str, List[float]], the metrics
        """
        history: Dict[str, List[float]] = defaultdict(list)
        n = x.shape[0]
        num_batches = int(np.ceil(n / batch_size))

        for epoch in range(epochs):
            epoch_metrics: Dict[str, List[float]] = defaultdict(list)

            for b in range(num_batches):
                s = b * batch_size
                e = min((b + 1) * batch_size, n)
                xb, yb = x[s:e], y[s:e]

                batch_stats = self.batch_step(xb, yb, training=True)
                print_stats(batch_stats, batch_num=b, num_batches=num_batches, epoch=epoch, avg=False)

                for k, v in batch_stats.items():
                    epoch_metrics[k].append(v)

            print_stats(epoch_metrics, epoch=epoch, avg=True)  # show epoch averages
            update_metric_dict(history, epoch_metrics)         # store epoch averages in history

        return history

    def evaluate(self, x: Tensor, y: Union[Tensor, np.ndarray], batch_size: int) -> Tuple[Dict[str, float], np.ndarray]:
        """
        X is the dataset inputs, Y is the dataset labels.
        Evaluates the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method. At the end, the metrics are returned. Should be called on
        the testing set to evaluate accuracy of the model using the metrics output from the fit method.

        Args:
            x: Tensor, the dataset inputs
            y: Tensor, the dataset labels
            batch_size: int, the batch size

        Returns:
            Tuple[Dict[str, float], np.ndarray], the metrics and predictions

        """
        n = x.shape[0]
        num_batches = int(np.ceil(n / batch_size))

        metrics_accum: Dict[str, List[float]] = defaultdict(list)
        preds_list: List[np.ndarray] = []

        for b in range(num_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n)
            xb, yb = x[s:e], y[s:e]

            batch_stats, batch_preds = self.batch_step(xb, yb, training=False)
            print_stats(batch_stats, batch_num=b, num_batches=num_batches, avg=False)

            for k, v in batch_stats.items():
                metrics_accum[k].append(v)
            preds_list.append(np.asarray(batch_preds))

        print_stats(metrics_accum, avg=True)
        final_metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
        predictions = np.concatenate(preds_list, axis=0) if preds_list else np.array([])

        return final_metrics, predictions

    def get_input_gradients(self) -> List[Tensor]:
        return super().get_input_gradients()

    def get_weight_gradients(self) -> List[Tensor]:
        return super().get_weight_gradients()
    
    @abstractmethod
    def batch_step(self, x: Tensor, y: Tensor, training: bool = True) -> Union[Dict[str, float], Tuple[Dict[str, float], Tensor]]:
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        raise NotImplementedError("batch_step method must be implemented in child class")

class SequentialModel(Model):
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass in sequential model. It's helpful to note that layers are initialized in beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_step(self, x: Tensor, y: Tensor, training: bool = True) -> Union[Dict[str, float], Tuple[Dict[str, float], Tensor]]:
        """Runs one forward pass; if training, also runs backprop and applies gradients."""
        with GradientTape() as tape:
            preds = self.forward(x)
            loss  = self.compiled_loss(preds, y)
            acc   = self.compiled_acc(preds, y)

        if training:
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(self.trainable_variables, grads)
            return {"loss": float(loss), "acc": float(acc)}
        else:
            return {"loss": float(loss), "acc": float(acc)}, preds
