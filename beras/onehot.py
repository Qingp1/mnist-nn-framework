import numpy as np
from typing import Dict

from beras.core import Callable, Tensor


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data: np.ndarray) -> None:
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        # Get unique labels
        unique_labels = np.unique(data)

        # Identity matrix, each row = one-hot vector
        eye = np.eye(len(unique_labels), dtype=np.float32)
        self.label_to_onehot: Dict = {
            label: eye[i] for i, label in enumerate(unique_labels)
        }


    def forward(self, data: np.ndarray) -> np.ndarray:
        # If encoder hasn’t been fit yet, fit it on this data
        if not hasattr(self, "label_to_onehot"):
            self.fit(data)

        # Convert each label into its one-hot vector
        encoded = np.array([self.label_to_onehot[label] for label in data])

        return encoded

    def inverse(self, data: np.ndarray) -> np.ndarray:
        # Get index of the "hot" position in each row
        indices = np.argmax(data, axis=1)

        # Map index → label using the keys from label_to_onehot
        index_to_label = {i: label for i, label in enumerate(self.label_to_onehot.keys())}

        return np.array([index_to_label[i] for i in indices])
