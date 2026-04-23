# mnist-nn-framework

A deep learning library built from scratch in Python, without using TensorFlow, PyTorch, or any ML frameworks. Trained on MNIST digit classification achieving 95%+ accuracy.

## Project Structure
```
beras/
├── core.py          # Base classes and autograd engine
├── layers.py        # Dense layers
├── activations.py   # ReLU, LeakyReLU, Sigmoid, Softmax
├── losses.py        # Cross-Entropy, MSE
├── metrics.py       # Accuracy metrics
├── optimizers.py    # Adam, RMSProp, SGD
├── gradient_tape.py # Custom gradient tape
└── onehot.py        # One-hot encoding
assignment.py        # Model definition and training
preprocess.py        # MNIST data loading and preprocessing
visualize.py         # Training visualization
```

## Requirements
```
pip install numpy
```

## How to Run
```
python assignment.py
```

## Results
- 95%+ accuracy on MNIST test set
- Trained using custom Adam optimizer and gradient tape
