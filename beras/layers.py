import numpy as np

from typing import Literal, List, Tuple
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size: int, output_size: int, initializer: DENSE_INITIALIZERS = "normal") -> None:
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> List[Tensor]:
        return [self.w, self.b]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

    def get_input_gradients(self) -> List[Tensor]:
        return [Tensor(self.w.T)]

    def get_weight_gradients(self) -> List[Tensor]:
        x = self.inputs[0]                      # (B, in_dim)
        B, in_dim = x.shape
        out_dim = self.w.shape[1]

        # Jacobian wrt W: (B, in_dim, out_dim)
        grad_w = x[:, :, None] * np.ones((B, 1, out_dim), dtype=x.dtype)

        # Jacobian wrt b: identity per sample  -> (B, out_dim, out_dim)
        grad_b = np.tile(np.eye(out_dim, dtype=x.dtype)[None, :, :], (B, 1, 1))

        return [Tensor(grad_w), Tensor(grad_b)]

    @staticmethod
    def _initialize_weight(initializer: DENSE_INITIALIZERS, input_size: int, output_size: int) -> Tuple[Variable, Variable]:
        """
        Initializes weight matrix and zero bias for a dense layer.

        Strategies:
            - zero: all zeros (symmetric, rarely useful in practice)
            - normal: standard normal N(0, 1)
            - xavier: scaled for tanh/sigmoid — std = sqrt(2 / (fan_in + fan_out))
            - kaiming: scaled for ReLU — std = sqrt(2 / fan_in)
            - xavier/kaiming uniform: uniform variants of the above
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"
        if initializer == "zero":
            w = np.zeros((input_size, output_size))
        elif initializer == "normal":
            w = np.random.randn(input_size, output_size)  # N(0,1)
        elif initializer == "xavier":
            std = np.sqrt(2.0 / (input_size + output_size))
            w = np.random.randn(input_size, output_size) * std
        elif initializer == "kaiming":
            std = np.sqrt(2.0 / input_size)
            w = np.random.randn(input_size, output_size) * std
        elif initializer == "xavier uniform":
            limit = np.sqrt(6.0 / (input_size + output_size))
            w = np.random.uniform(-limit, limit, (input_size, output_size))
        elif initializer == "kaiming uniform":
            limit = np.sqrt(6.0 / input_size)
            w = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        b = np.zeros((1, output_size))  # bias shape (1, output_size)

        return Variable(w), Variable(b)