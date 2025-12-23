# MNIST Neural Network Implementation

A modern, well-engineered C++ implementation of a neural network for MNIST handwritten digit recognition.

## Features

- Complete neural network implementation from scratch
- Supports various activation functions: ReLU, Sigmoid, Tanh, Softmax
- Multiple loss functions: MSE, CrossEntropy, CategoricalCrossEntropy
- Optimizers: SGD, SGD with Momentum, Adam
- Training process with callbacks: console logging, early stopping, checkpointing
- Clean, modern C++ code (C++23)
- Well-documented with detailed comments

## Project Structure

```
mnist/
├── include/             # Header files
│   ├── activation.h     # Activation functions
│   ├── dataset.h        # MNIST dataset loader
│   ├── layer.h          # Neural network layers
│   ├── loss.h           # Loss functions
│   ├── network.h        # Neural network model
│   ├── optimizer.h      # Optimization algorithms
│   └── trainer.h        # Training utilities
├── src/                 # Implementation files
│   ├── activation.cpp
│   ├── dataset.cpp
│   ├── layer.cpp
│   ├── loss.cpp
│   ├── main.cpp         # Main application
│   ├── network.cpp
│   ├── optimizer.cpp
│   └── trainer.cpp
└── data/                # Directory for MNIST data files
```

## Dependencies

- C++23 compiler (GCC 10+, Clang 10+, or MSVC 19.29+)
- Eigen3 for matrix operations
- Meson build system

## Building the Project

1. Ensure you have Meson and Ninja installed:
   ```
   pip install meson ninja
   ```

2. Clone the repository and navigate to the project directory:
   ```
   cd /path/to/hahaha/sample
   ```

3. Configure and build the project:
   ```
   meson setup builddir
   cd builddir
   ninja
   ```

## Downloading the MNIST Dataset

The MNIST dataset needs to be downloaded and extracted to the `data` directory. The dataset consists of four files:

- `train-images-idx3-ubyte`: Training images
- `train-labels-idx1-ubyte`: Training labels
- `t10k-images-idx3-ubyte`: Test images
- `t10k-labels-idx1-ubyte`: Test labels

You can download these files from Yann LeCun's website:
http://yann.lecun.com/exdb/mnist/

## Running the Application

After building, you can run the MNIST recognizer:

```
./mnist_recognizer --data-dir=/path/to/mnist/data
```

### Command Line Options

- `--data-dir=PATH`: Path to the MNIST data directory
- `--epochs=N`: Number of training epochs (default: 10)
- `--batch-size=N`: Batch size for training (default: 64)
- `--learning-rate=N`: Initial learning rate (default: 0.001)
- `--checkpoint-dir=PATH`: Directory to save model checkpoints (default: none)
- `--load-model=PATH`: Path to load a pre-trained model (default: none)
- `--save-model=PATH`: Path to save the final model (default: none)
- `--help`: Display help message and exit

## Example Usage

Train a new model:
```
./mnist_recognizer --data-dir=../data --epochs=20 --batch-size=128 --save-model=model.bin
```

Load a pre-trained model and test:
```
./mnist_recognizer --data-dir=../data --load-model=model.bin
```

## Implementation Details

### Neural Network Architecture

The default architecture used is:
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation

### Training Process

The training process includes:
- Mini-batch gradient descent
- Categorical cross-entropy loss function
- Adam optimizer with learning rate of 0.001
- Early stopping to prevent overfitting
- Optional model checkpointing

## Customizing the Network

You can customize the network architecture and training parameters by modifying the `main.cpp` file.
