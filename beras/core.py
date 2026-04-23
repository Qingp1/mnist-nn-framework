from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
 
from typing import TYPE_CHECKING, TypedDict, Dict, Union, Any, Optional, List, Iterable, Tuple
if TYPE_CHECKING:
    from beras.gradient_tape import GradientTape


class Tensor(np.ndarray):
    """NumPy array subclass that tracks whether it is trainable."""
    def __new__(cls, input_array: Union[np.ndarray, List, Tuple, 'Tensor']) -> 'Tensor':
        if isinstance(input_array, Tensor):
            return input_array
        obj = np.asarray(a=input_array).view(type=cls)
        obj.trainable = True
        return obj

    def __array_finalize__(self, obj: Optional[Any]) -> None:
        if obj is None:
            return
        self.trainable = getattr(obj, "trainable", True)

    def assign(self, value: Union[Tensor, np.ndarray]) -> None:
        self[:] = value

"""
Mimics the tf.Variable class.
"""
class Variable(Tensor):

    def assign(self, value: Union[Tensor, np.ndarray]) -> None:
        self[:] = value

class Callable(ABC):
    """Modules that can be called like functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Delegates to forward(); overridden by Diffable to also record gradients."""
        return Tensor(self.forward(*args, **kwargs))

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        pass


class Weighted(ABC):
    """
    Modules that have weights.
    """

    @property
    @abstractmethod
    def weights(self) -> List[Tensor]:
        pass

    @property
    def trainable_variables(self) -> List[Tensor]:
        """Collects all trainable variables in the module"""
        return [w for w in self.weights if getattr(w, "trainable", False)]

    @property
    def non_trainable_variables(self) -> List[Tensor]:
        """Collects all non-trainable variables in the module"""
        return [w for w in self.weights if not getattr(w, "trainable", False)]

    @property
    def trainable(self) -> bool:
        """Returns true if any of the weights are trainable"""
        return any(getattr(w, "trainable", False) for w in self.weights)

    @trainable.setter
    def trainable(self, trainable: bool) -> None:
        """Sets the trainable status of all weights to trainable"""
        for w in self.weights:
            w.trainable = trainable


class Diffable(Callable, Weighted):
    """
    Modules that keep track of gradients
    """

    # We define gradient tape as a class variable so that it can be accessed from anywhere
    #  and so that it can be set to None when we don't want to record gradients
    gradient_tape: Optional['GradientTape'] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Tensor, List[Tensor]]:
        """
        Runs forward pass and, if a GradientTape scope is active, registers this layer
        as the producer of each output so gradients can be traced back through it.
        """
        # Capture named inputs by matching positional args to forward()'s parameter names
        self.argnames = self.forward.__code__.co_varnames[1:]
        named_args = {self.argnames[i]: args[i] for i in range(len(args))}
        self.input_dict = {**named_args, **kwargs}
        self.inputs = [
            self.input_dict[arg]
            for arg in self.argnames
            if arg in self.input_dict.keys()
        ]

        self.outputs = self.forward(*args, **kwargs)

        list_outs = isinstance(self.outputs, list) or isinstance(self.outputs, tuple)
        if not list_outs:
            self.outputs = [self.outputs]

        if Diffable.gradient_tape is not None:
            for out in self.outputs:
                Diffable.gradient_tape.previous_layers[id(out)] = self

        return self.outputs if list_outs else self.outputs[0]

    @abstractmethod
    def get_input_gradients(self) -> List[Tensor]:
        """
        NOTE: required for all Diffable modules
        returns:
            list of gradients with respect to the inputs
        """
        return []

    @abstractmethod
    def get_weight_gradients(self) -> List[Tensor]:
        """
        NOTE: required for SOME Diffable modules
        returns:
            list of gradients with respect to the weights
        """
        return []

    def compose_input_gradients(self, J: Optional[List[Tensor]] = None) -> List[Tensor]:
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied

        Note: Generalized to handle a wide array of architectures, closely mirroring
                how this works in TensorFlow.
        """
        # If J[0] is None, then we have no upstream gradients to compose with
        #  so we just return the input gradients
        # if J is None or J[0] is None:
        if J is None or J[0] is None:
            return self.get_input_gradients()
        # J_out stores all input gradients to be tracked in backpropagation.
        J_out = []
        for upstream_jacobian in J:
            batch_size = upstream_jacobian.shape[0]
            for layer_input, inp_grad in zip(self.inputs, self.get_input_gradients()):
                j_wrt_lay_inp = np.zeros(layer_input.shape, dtype=inp_grad.dtype)
                for sample in range(batch_size):
                    s_grad = inp_grad[sample] if len(inp_grad.shape) == 3 else inp_grad
                    try:
                        j_wrt_lay_inp[sample] = s_grad @ upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b @ j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                J_out += [j_wrt_lay_inp]
        # Returns cumulative jacobians w.r.t to all inputs.
        return J_out

    def compose_weight_gradients(self, J: Optional[List[Tensor]] = None) -> List[Tensor]:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        # Returns weights gradients if no apriori cumulative jacobians are provided.
        if J is None or J[0] is None:
            return self.get_weight_gradients()
        # J_out stores all weight gradients to be tracked in further backpropagation.
        J_out = []
        ## For every weight/weight-gradient pair...
        for upstream_jacobian in J:
            for layer_w, w_grad in zip(self.weights, self.get_weight_gradients()):
                batch_size = upstream_jacobian.shape[0]
                ## Make a cumulative jacobian which will contribute to the final jacobian
                j_wrt_lay_w = np.zeros((batch_size, *layer_w.shape), dtype=w_grad.dtype)
                ## For every element in the batch (for a single batch-level gradient updates)
                for sample in range(batch_size):
                    ## If the weight gradient is a batch of transform matrices, get the right entry.
                    ## Allows gradient methods to give either batched or non-batched matrices
                    s_grad = w_grad[sample] if len(w_grad.shape) == 3 else w_grad
                    ## Update the batch's Jacobian update contribution
                    try:
                        j_wrt_lay_w[sample] = s_grad * upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b * j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                ## The final jacobian for this weight is the average gradient update for the batch
                J_out += [np.sum(j_wrt_lay_w, axis=0)]
            ## After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out
