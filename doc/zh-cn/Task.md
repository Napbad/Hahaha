# Hahaha Project Task List: 全能型数学与人工智能库

Hahaha 项目旨在构建一个集数学计算（类似 NumPy）、机器学习（类似 Scikit-Learn）与深度学习（类似 PyTorch）于一体的 C++ 核心库。

---

## 1. 数学计算模块 (Math & Numerical Computing) - [NumPy 核心]

提供高性能的多维张量操作及数学工具。

- [x] **张量核心基础设施 (Tensor Infrastructure)**
    - [x] `TensorShape`: 维度管理。
    - [x] `TensorStride`: 内存布局计算。
    - [x] `TensorData`: CPU 内存分配与所有权管理。
    - [x] `NestedData`: 支持多层嵌套初始化。
    - [x] `ScalarTensor` 广播适配。
- [ ] **线性代数 (Linear Algebra)**
    - [x] `matmul`: 矩阵乘法。
    - [x] `transpose`: 2D 转置。
    - [ ] `dot`: 向量点积。
    - [ ] `inverse`: 逆矩阵计算。
    - [ ] `det`: 行列式。
    - [ ] `eigen`: 特征值与特征向量。
    - [ ] `SVD`: 奇异值分解。
    - [ ] `QR`: QR 分解。
- [ ] **统计与函数 (Statistics & Math Functions)**
    - [x] `sum`: 求和。
    - [ ] `mean`, `var`, `std`: 均值、方差、标准差。
    - [ ] `max`, `min`, `argmax`, `argmin`: 极值与索引。
    - [ ] `exp`, `log`, `abs`, `sqrt`: 元素级数学函数。
- [ ] **张量操作高级 API (Advanced Manipulation)**
    - [x] `reshape`: 形状变换。
    - [ ] `slice/crop`: 张量切片与裁剪（如 `a[{0, 5}, {2, 4}]`）。
    - [ ] `concatenate/stack`: 张量拼接。
    - [ ] `split`: 张量拆分。
    - [ ] `broadcast_to`: 显式广播。
- [ ] **随机数生成 (Random)**
    - [ ] `uniform`, `normal`: 均匀分布与正态分布采样。
    - [ ] `shuffle`: 乱序。

---

## 2. 机器学习模块 (Machine Learning) - [Scikit-Learn 核心]

提供经典的机器学习算法及评价工具。

- [ ] **数据预处理 (Preprocessing)**
    - [ ] `StandardScaler`: 标准化（均值 0，方差 1）。
    - [ ] `MinMaxScaler`: 归一化（0 到 1）。
    - [ ] `OneHotEncoder`: 独热编码。
- [ ] **经典回归与分类算法 (Classical Algorithms)**
    - [ ] `LinearRegression`: 线性回归。
    - [ ] `LogisticRegression`: 逻辑回归。
    - [ ] `DecisionTree`: 决策树。
    - [ ] `SVM`: 支持向量机。
    - [ ] `KMeans`: K-均值聚类。
    - [ ] `PCA`: 主成分分析。
- [ ] **模型评估 (Model Evaluation & Metrics)**
    - [ ] `Accuracy`, `F1-Score`, `Precision`, `Recall`。
    - [ ] `MSE`, `MAE`, `R2-Score`。
    - [ ] `ConfusionMatrix`: 混淆矩阵。
- [ ] **训练辅助 (Training Helpers)**
    - [ ] `train_test_split`: 数据集切分。
    - [ ] `cross_val_score`: 交叉验证。

---

## 3. 深度学习模块 (Deep Learning & Autograd) - [PyTorch 核心]

提供基于计算图的自动微分及神经网络组件。

- [x] **自动微分引擎 (Autograd Engine)**
    - [x] `ComputeNode`: 动态计算图节点。
    - [x] `TopoSort`: 拓扑排序算法。
    - [x] `backward`: 自动反向传播。
    - [x] `Gradient Accumulation`: 梯度累加。
- [ ] **神经网络层 (NN Layers)**
    - [ ] `Linear (Dense)`: 全连接层。
    - [ ] `Conv2D`: 卷积层。
    - [ ] `MaxPool2D`, `AvgPool2D`: 池化层。
    - [ ] `BatchNorm`: 批标准化。
    - [ ] `Dropout`: 随机失活层。
    - [ ] `RNN / LSTM / GRU`: 循环神经网络。
- [ ] **激活函数 (Activations)**
    - [ ] `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`。
- [ ] **损失函数 (Loss Functions)**
    - [ ] `MSELoss`, `CrossEntropyLoss`, `BCELoss`。
- [x] **优化器 (Optimizers)**
    - [x] `Optimizer` 基类。
    - [x] `SGD`: 随机梯度下降。
    - [ ] `Adam`, `AdamW`, `RMSProp`。
- [ ] **高级抽象 (DL Abstraction)**
    - [ ] `Module`: 模型基类（类似 `nn.Module`）。
    - [ ] `Dataset` / `DataLoader`: 高效数据读取与批处理。

---

## 4. 后端与性能加速 (Backend & Optimization)

- [x] **多设备框架 (Device Framework)**
    - [x] `DeviceComputeDispatcher`: 任务分发器。
    - [x] `Device` 对象管理。
- [ ] **SIMD 向量化 (SIMD Acceleration)**
    - [ ] AVX2 / AVX-512 (Intel/AMD)。
    - [ ] NEON (ARM)。
- [ ] **GPU/CUDA 加速 (CUDA Backend)**
    - [ ] `GpuMemory`: CUDA 显存管理。
    - [ ] 高性能 CUDA Kernels 实现（加、减、乘、除、卷积、矩阵乘法）。
    - [ ] 半精度 (FP16) 与混合精度训练。

---

## 5. 可视化、工具与生态 (Tools & Ecosystem)

- [x] **实时可视化 (Visualizer)**
    - [x] 基于 ImGui 的训练曲线实时绘制。
    - [x] 实时参数分布展示。
- [x] **日志与监控 (Logging)**
    - [x] 线程安全的彩色日志系统。
- [ ] **序列化 (Serialization)**
    - [ ] 模型保存与加载 (ONNX 兼容）。
- [ ] **跨语言支持 (Bindings)**
    - [ ] Python 绑定 (Pybind11)，支持 `import hahaha`。

---

## 6. 应用示例 (Examples)

- [ ] **MNIST**: 手写数字识别（深度学习示例）。
- [ ] **Iris**: 鸢尾花分类（传统机器学习示例）。
- [ ] **Linear Regression Demo**: 房价预测（数学回归示例）。
- [ ] **Matrix Benchmark**: 与 NumPy/Eigen 的性能对比。
