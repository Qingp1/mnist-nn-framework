import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Any

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self) -> None:
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: Dict[int, Optional[Diffable]] = defaultdict(lambda: None)

    def __enter__(self) -> 'GradientTape':
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: List[Tensor]) -> List[Tensor]:
        """
        Computes gradients of `target` with respect to each tensor in `sources`
        via reverse-mode autodiff (BFS over the recorded computation graph).
        """
        # BFS queue; grads maps id(tensor) -> list of gradient contributions
        queue = [target]
        grads = defaultdict(list)

        target_array = np.array(target)
        if target_array.ndim == 0:
            initial_grad = np.ones((1,))
        else:
            initial_grad = np.ones_like(target_array)
        grads[id(target)].append(initial_grad)

        while queue:
            tensor = queue.pop(0)
            layer = self.previous_layers[id(tensor)]
            if layer is None:
                continue

            grad_list = grads[id(tensor)]
            
            # For scalar target (loss), create upstream gradient matching batch size
            if tensor is target and np.array(tensor).ndim == 0:
                # Get batch size from layer inputs
                batch_size = layer.inputs[0].shape[0]
                upstream = np.ones((batch_size, 1))
            elif len(grad_list) == 1:
                upstream = grad_list[0]
            else:
                upstream = np.sum(grad_list, axis=0)

            # grads wrt inputs
            input_grads = layer.compose_input_gradients([upstream])
            for inp, g in zip(layer.inputs, input_grads):
                grads[id(inp)].append(g)
                queue.append(inp)

            # grads wrt weights
            weight_grads = layer.compose_weight_gradients([upstream])
            for w, g in zip(layer.weights, weight_grads):
                grads[id(w)].append(np.array(g))

        results = []                        # final gradients for sources
        for src in sources:
            g_list = grads[id(src)]
            if not g_list:
                results.append(None)
            else:
                if len(g_list) == 1:
                    total_grad = g_list[0]
                else:
                    total_grad = np.sum(g_list, axis=0)
                results.append(Tensor(total_grad))
        return results
