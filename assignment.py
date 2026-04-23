from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
from visualize import visualize_predictions
import numpy as np

from beras.model import SequentialModel

def get_model() -> SequentialModel:
    model = SequentialModel(
        [
            # 28x28 input flattened to 784 features
            Dense(784, 256, initializer="kaiming"),
            ReLU(),
            Dense(256, 128, initializer="kaiming"),
            ReLU(),
            Dense(128, 10, initializer="xavier"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    return Adam(learning_rate=1e-3)

def get_loss_fn():
    return CategoricalCrossEntropy(epsilon=1e-12)

def get_acc_fn():
    return CategoricalAccuracy()

if __name__ == '__main__':
    model = get_model()

    model.compile(
        optimizer=get_optimizer(),
        loss_fn=get_loss_fn(),
        acc_fn=get_acc_fn()
    )

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    ohe = OneHotEncoder(10)
    y_train_oh = ohe(y_train)
    y_test_oh = ohe(y_test)

    history = model.fit(x_train, y_train_oh, epochs=10, batch_size=128)

    metrics, preds = model.evaluate(x_test, y_test_oh, batch_size=256)
    print("\nTest metrics:", metrics)

    visualize_predictions(x_test, y_test_oh, preds)

