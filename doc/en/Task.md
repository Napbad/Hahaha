# Hahaha Project Task List: Versatile Math and AI Library

The Hahaha project aims to build an all-in-one C++ core library that integrates Numerical Computing (like NumPy), Machine Learning (like Scikit-Learn), and Deep Learning (like PyTorch).

---

## 1. Math & Numerical Computing - [NumPy Core]

Provides high-performance multi-dimensional tensor operations and mathematical tools.

- [x] **Tensor Core Infrastructure**
    - [x] `TensorShape`: Dimension management.
    - [x] `TensorStride`: Memory layout calculation.
    - [x] `TensorData`: CPU memory allocation and ownership management.
    - [x] `NestedData`: Support for multi-level nested initialization.
    - [x] `ScalarTensor` broadcasting adaptation.
- [ ] **Linear Algebra**
    - [x] `matmul`: Matrix multiplication.
    - [x] `transpose`: 2D transposition.
    - [ ] `dot`: Vector dot product.
    - [ ] `inverse`: Matrix inversion.
    - [ ] `det`: Determinant.
    - [ ] `eigen`: Eigenvalues and eigenvectors.
    - [ ] `SVD`: Singular Value Decomposition.
    - [ ] `QR`: QR Decomposition.
- [ ] **Statistics & Math Functions**
    - [x] `sum`: Global summation.
    - [ ] `mean`, `var`, `std`: Mean, variance, and standard deviation.
    - [ ] `max`, `min`, `argmax`, `argmin`: Extrema and indices.
    - [ ] `exp`, `log`, `abs`, `sqrt`: Element-wise mathematical functions.
- [ ] **Advanced Tensor Manipulation**
    - [x] `reshape`: Shape transformation.
    - [ ] `slice/crop`: Tensor slicing and cropping (e.g., `a[{0, 5}, {2, 4}]`).
    - [ ] `concatenate/stack`: Tensor joining.
    - [ ] `split`: Tensor splitting.
    - [ ] `broadcast_to`: Explicit broadcasting.
- [ ] **Random Number Generation**
    - [ ] `uniform`, `normal`: Sampling from uniform and normal distributions.
    - [ ] `shuffle`: Random shuffling.

---

## 2. Machine Learning - [Scikit-Learn Core]

Provides classical machine learning algorithms and evaluation tools.

- [ ] **Data Preprocessing**
    - [ ] `StandardScaler`: Standardization (mean 0, variance 1).
    - [ ] `MinMaxScaler`: Normalization (0 to 1).
    - [ ] `OneHotEncoder`: One-hot encoding.
- [ ] **Classical Regression & Classification Algorithms**
    - [ ] `LinearRegression`: Linear regression.
    - [ ] `LogisticRegression`: Logistic regression.
    - [ ] `DecisionTree`: Decision trees.
    - [ ] `SVM`: Support Vector Machines.
    - [ ] `KMeans`: K-Means clustering.
    - [ ] `PCA`: Principal Component Analysis.
- [ ] **Model Evaluation & Metrics**
    - [ ] `Accuracy`, `F1-Score`, `Precision`, `Recall`.
    - [ ] `MSE`, `MAE`, `R2-Score`.
    - [ ] `ConfusionMatrix`: Confusion matrix.
- [ ] **Training Helpers**
    - [ ] `train_test_split`: Dataset splitting.
    - [ ] `cross_val_score`: Cross-validation.

---

## 3. Deep Learning & Autograd - [PyTorch Core]

Provides computational graph-based automatic differentiation and neural network components.

- [x] **Autograd Engine**
    - [x] `ComputeNode`: Dynamic computational graph nodes.
    - [x] `TopoSort`: Topological sorting algorithm.
    - [x] `backward`: Automatic backpropagation.
    - [x] `Gradient Accumulation`: Gradient summing for shared nodes.
- [ ] **Neural Network Layers (NN Layers)**
    - [ ] `Linear (Dense)`: Fully connected layer.
    - [ ] `Conv2D`: Convolutional layer.
    - [ ] `MaxPool2D`, `AvgPool2D`: Pooling layers.
    - [ ] `BatchNorm`: Batch Normalization.
    - [ ] `Dropout`: Random dropout layer.
    - [ ] `RNN / LSTM / GRU`: Recurrent neural networks.
- [ ] **Activations**
    - [ ] `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`.
- [ ] **Loss Functions**
    - [ ] `MSELoss`, `CrossEntropyLoss`, `BCELoss`.
- [x] **Optimizers**
    - [x] `Optimizer` base class.
    - [x] `SGD`: Stochastic Gradient Descent.
    - [ ] `Adam`, `AdamW`, `RMSProp`.
- [ ] **High-level Abstractions**
    - [ ] `Module`: Model base class (similar to `nn.Module`).
    - [ ] `Dataset` / `DataLoader`: Efficient data loading and batching.

---

## 4. Backend & Optimization

- [x] **Device Framework**
    - [x] `DeviceComputeDispatcher`: Task dispatcher.
    - [x] `Device` object management.
- [ ] **SIMD Acceleration**
    - [ ] AVX2 / AVX-512 (Intel/AMD).
    - [ ] NEON (ARM).
- [ ] **GPU/CUDA Acceleration**
    - [ ] `GpuMemory`: CUDA memory management.
    - [ ] High-performance CUDA Kernels (Add, Sub, Mul, Div, Conv, MatMul).
    - [ ] Mixed-precision training (FP16).

---

## 5. Tools & Ecosystem

- [x] **Real-time Visualization**
    - [x] Real-time training curves using ImGui.
    - [x] Parameter distribution display.
- [x] **Logging**
    - [x] Thread-safe colored logging system.
- [ ] **Serialization**
    - [ ] Model saving and loading (ONNX compatible).
- [ ] **Language Bindings**
    - [ ] Python bindings (Pybind11), supporting `import hahaha`.

---

## 6. Application Examples

- [ ] **MNIST**: Handwritten digit recognition (Deep Learning example).
- [ ] **Iris**: Flower classification (Classical ML example).
- [ ] **Linear Regression Demo**: House price prediction (Math/Regression example).
- [ ] **Matrix Benchmark**: Performance comparison with NumPy/Eigen.

