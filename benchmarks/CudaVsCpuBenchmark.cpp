// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "backend/Device.h"
#include "backend/DeviceComputeDispatcher.h"
#include "backend/cpu/CPUDevice.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#endif
#endif

using h3::backend::DeviceType;
using h3::math::NestedData;
using h3::math::TensorShape;
using h3::math::TensorWrapper;

/**
 * @brief Simple timer utility for benchmarking.
 */
class Timer {
  public:
    void start() {
        startTime_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedMilliseconds() const {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime_);
        return duration.count() / 1000.0; // Convert to milliseconds
    }

  private:
    std::chrono::high_resolution_clock::time_point startTime_;
};

/**
 * @brief Benchmark a tensor operation and return average time in milliseconds.
 */
template <typename Func>
double benchmarkOperation(Func&& func, int iterations = 10) {
    Timer timer;
    double totalTime = 0.0;

    // Warm-up
    for (int i = 0; i < 3; ++i) {
        func();
    }

    // Actual benchmark
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        func();
        totalTime += timer.elapsedMilliseconds();
    }

    return totalTime / iterations;
}

/**
 * @brief Create a tensor with random data.
 */
TensorWrapper<float> createRandomTensor(const TensorShape& shape) {
    size_t totalSize = 1;
    for (size_t dim : shape.getDims()) {
        totalSize *= dim;
    }

    // Create tensor with correct shape directly
    TensorWrapper<float> tensor(shape, 0.0f);

    // Fill with random data using raw data access (more efficient)
    auto& rawData = tensor.getRawData();
    for (size_t i = 0; i < totalSize; ++i) {
        rawData[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    return tensor;
}

/**
 * @brief Benchmark addition operation on CPU.
 */
double benchmarkAddCPU(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);
    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    a.to(cpuDevice);
    b.to(cpuDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cpuDevice);
            auto res = hahaha::backend::dispatchAdd<float>(
                DeviceType::CPU, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark subtraction operation on CPU.
 */
double benchmarkSubCPU(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);
    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    a.to(cpuDevice);
    b.to(cpuDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cpuDevice);
            auto res = hahaha::backend::dispatchSub<float>(
                DeviceType::CPU, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark multiplication operation on CPU.
 */
double benchmarkMulCPU(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);
    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    a.to(cpuDevice);
    b.to(cpuDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cpuDevice);
            auto res = hahaha::backend::dispatchMul<float>(
                DeviceType::CPU, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark division operation on CPU.
 */
double benchmarkDivCPU(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);
    // Avoid division by zero - modify tensor directly
    auto& bRawData = b.getRawData();
    for (size_t i = 0; i < b.getTotalSize(); ++i) {
        if (std::abs(bRawData[i]) < 0.1f) {
            bRawData[i] = (bRawData[i] >= 0) ? 0.1f : -0.1f;
        }
    }
    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    a.to(cpuDevice);
    b.to(cpuDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cpuDevice);
            auto res = hahaha::backend::dispatchDiv<float>(
                DeviceType::CPU, a, b, result);
            return res;
        },
        iterations);
}

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
/**
 * @brief Benchmark addition operation on CUDA.
 */
double benchmarkAddCUDA(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    auto cudaDevice = std::make_shared<hahaha::backend::CudaDevice>(&prop);
    a.to(cudaDevice);
    b.to(cudaDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cudaDevice);
            auto res = hahaha::backend::dispatchAdd<float>(
                DeviceType::CUDA, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark subtraction operation on CUDA.
 */
double benchmarkSubCUDA(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    auto cudaDevice = std::make_shared<hahaha::backend::CudaDevice>(&prop);
    a.to(cudaDevice);
    b.to(cudaDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cudaDevice);
            auto res = hahaha::backend::dispatchSub<float>(
                DeviceType::CUDA, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark multiplication operation on CUDA.
 */
double benchmarkMulCUDA(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    auto cudaDevice = std::make_shared<hahaha::backend::CudaDevice>(&prop);
    a.to(cudaDevice);
    b.to(cudaDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cudaDevice);
            auto res = hahaha::backend::dispatchMul<float>(
                DeviceType::CUDA, a, b, result);
            return res;
        },
        iterations);
}

/**
 * @brief Benchmark division operation on CUDA.
 */
double benchmarkDivCUDA(const TensorShape& shape, int iterations = 10) {
    auto a = createRandomTensor(shape);
    auto b = createRandomTensor(shape);
    // Avoid division by zero - create new tensor with safe values
    std::vector<float> bData;
    for (size_t i = 0; i < b.getTotalSize(); ++i) {
        float val = b.at({i});
        if (std::abs(val) < 0.1f) {
            bData.push_back((val >= 0) ? 0.1f : -0.1f);
        } else {
            bData.push_back(val);
        }
    }
    TensorWrapper<float> bSafe(bData);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    auto cudaDevice = std::make_shared<hahaha::backend::CudaDevice>(&prop);
    a.to(cudaDevice);
    b.to(cudaDevice);

    return benchmarkOperation(
        [&]() {
            auto result = TensorWrapper<float>(shape, 0.0f);
            result.to(cudaDevice);
            auto res = hahaha::backend::dispatchDiv<float>(
                DeviceType::CUDA, a, bSafe, result);
            return res;
        },
        iterations);
}
#endif
#endif

/**
 * @brief Print benchmark results in a formatted table.
 */
void printBenchmarkResults(const std::string& operation,
                           const TensorShape& shape,
                           double cpuTime,
                           double cudaTime,
                           bool cudaAvailable) {
    std::cout << "\n=== " << operation << " Benchmark ===" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.getDims().size(); ++i) {
        std::cout << shape.getDims()[i];
        if (i < shape.getDims().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Total elements: " << shape.getTotalSize() << std::endl;
    std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

    if (cudaAvailable) {
        std::cout << "CUDA time: " << cudaTime << " ms" << std::endl;
        double speedup = cpuTime / cudaTime;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        if (speedup > 1.0) {
            std::cout << "CUDA is " << speedup << "x faster" << std::endl;
        } else {
            std::cout << "CPU is " << (1.0 / speedup) << "x faster"
                      << std::endl;
        }
    } else {
        std::cout << "CUDA: Not available" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA vs CPU Benchmark Test" << std::endl;
    std::cout << "========================================" << std::endl;

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
    bool cudaAvailable = true;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaAvailable = false;
        std::cout << "Warning: CUDA device not available, only CPU benchmarks "
                     "will run."
                  << std::endl;
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "CUDA Compute Capability: " << prop.major << "."
                  << prop.minor << std::endl;
    }
#else
    bool cudaAvailable = false;
    std::cout << "CUDA headers not available, only CPU benchmarks will run."
              << std::endl;
#endif
#else
    bool cudaAvailable = false;
    std::cout << "CUDA not enabled, only CPU benchmarks will run." << std::endl;
#endif

    // Test different tensor sizes
    std::vector<TensorShape> shapes = {
        TensorShape({100}),          // Small
        TensorShape({1000}),         // Medium
        TensorShape({10000}),        // Large
        TensorShape({100000}),       // Very large
        TensorShape({1000, 1000}),   // 2D large
        TensorShape({100, 100, 100}) // 3D
    };

    const int iterations = 10;

    for (const auto& shape : shapes) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing shape: [";
        for (size_t i = 0; i < shape.getDims().size(); ++i) {
            std::cout << shape.getDims()[i];
            if (i < shape.getDims().size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "] (" << shape.getTotalSize() << " elements)" << std::endl;

        // Addition
        double cpuAdd = benchmarkAddCPU(shape, iterations);
        double cudaAdd = 0.0;
        if (cudaAvailable) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
            cudaAdd = benchmarkAddCUDA(shape, iterations);
#endif
#endif
        }
        printBenchmarkResults(
            "Addition", shape, cpuAdd, cudaAdd, cudaAvailable);

        // Subtraction
        double cpuSub = benchmarkSubCPU(shape, iterations);
        double cudaSub = 0.0;
        if (cudaAvailable) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
            cudaSub = benchmarkSubCUDA(shape, iterations);
#endif
#endif
        }
        printBenchmarkResults(
            "Subtraction", shape, cpuSub, cudaSub, cudaAvailable);

        // Multiplication
        double cpuMul = benchmarkMulCPU(shape, iterations);
        double cudaMul = 0.0;
        if (cudaAvailable) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
            cudaMul = benchmarkMulCUDA(shape, iterations);
#endif
#endif
        }
        printBenchmarkResults(
            "Multiplication", shape, cpuMul, cudaMul, cudaAvailable);

        // Division
        double cpuDiv = benchmarkDivCPU(shape, iterations);
        double cudaDiv = 0.0;
        if (cudaAvailable) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
            cudaDiv = benchmarkDivCUDA(shape, iterations);
#endif
#endif
        }
        printBenchmarkResults(
            "Division", shape, cpuDiv, cudaDiv, cudaAvailable);
    }

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Benchmark completed!" << std::endl;

    return 0;
}
