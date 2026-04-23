import numpy as np
from typing import Tuple
from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Loads MNIST, normalizes pixel values to [0, 1], flattens images to 1D, and wraps in Tensors."""
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()

    train_inputs = train_inputs.astype(np.float32) / 255.0
    test_inputs = test_inputs.astype(np.float32) / 255.0
    train_inputs = train_inputs.reshape(train_inputs.shape[0], -1)
    test_inputs = test_inputs.reshape(test_inputs.shape[0], -1)

    train_inputs = Tensor(train_inputs)
    train_labels = Tensor(train_labels)
    test_inputs = Tensor(test_inputs)
    test_labels = Tensor(test_labels)

    return train_inputs, train_labels, test_inputs, test_labels

