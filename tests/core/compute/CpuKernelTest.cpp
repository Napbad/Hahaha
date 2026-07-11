//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

/// Tests for CPU Kernels - element-wise operations

#include <gtest/gtest.h>
#include <cmath>

#define _USE_MATH_DEFINES
#include <cmath>

#include "compute/ComputeContextV2.h"
#include "compute/kernels/CpuKernels.h"
#include "backend/Device.h"
#include "defines.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class CpuKernelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    math::TensorInner createFloat32Tensor(const std::vector<Int64>& shape, Float32* data) {
        math::TensorMetadata meta;
        meta.dataType = DataType::Float32;
        meta.isContiguous = true;
        meta.isView = false;
        auto tensor = math::TensorInner(shape, meta);
        
        Float32* tensorData = tensor.storageRef().data().as<Float32>();
        for (size_t i = 0; i < tensor.shapeRef().getTotalSize(); ++i) {
            tensorData[i] = data[i];
        }
        return tensor;
    }
    
    bool compareFloats(Float32 a, Float32 b, Float32 epsilon = 1e-5f) {
        return std::fabs(a - b) < epsilon;
    }
};

// ============== Binary Operation Tests ==============

TEST_F(CpuKernelTest, AddKernelBasic) {
    std::vector<Float32> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> b = {5.0f, 4.0f, 3.0f, 2.0f};
    std::vector<Float32> expected = {6.0f, 6.0f, 6.0f, 6.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data()); // Will be overwritten
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::addKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, SubKernelBasic) {
    std::vector<Float32> a = {10.0f, 8.0f, 6.0f, 4.0f};
    std::vector<Float32> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> expected = {9.0f, 6.0f, 3.0f, 0.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::subKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, MulKernelBasic) {
    std::vector<Float32> a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<Float32> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> expected = {2.0f, 6.0f, 12.0f, 20.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::mulKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, DivKernelBasic) {
    std::vector<Float32> a = {10.0f, 9.0f, 8.0f, 6.0f};
    std::vector<Float32> b = {2.0f, 3.0f, 4.0f, 2.0f};
    std::vector<Float32> expected = {5.0f, 3.0f, 2.0f, 3.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::divKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, MaxKernelBasic) {
    std::vector<Float32> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<Float32> b = {4.0f, 2.0f, 7.0f, 6.0f};
    std::vector<Float32> expected = {4.0f, 5.0f, 7.0f, 8.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::maxKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, MinKernelBasic) {
    std::vector<Float32> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<Float32> b = {4.0f, 2.0f, 7.0f, 6.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 6.0f};
    
    auto tensorA = createFloat32Tensor({4}, a.data());
    auto tensorB = createFloat32Tensor({4}, b.data());
    auto tensorDst = createFloat32Tensor({4}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::minKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

// ============== Unary Operation Tests ==============

TEST_F(CpuKernelTest, SqrtKernelBasic) {
    std::vector<Float32> input = {1.0f, 4.0f, 9.0f, 16.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto tensorSrc = createFloat32Tensor({4}, input.data());
    auto tensorDst = createFloat32Tensor({4}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::sqrtKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, AbsKernelBasic) {
    std::vector<Float32> input = {-1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto tensorSrc = createFloat32Tensor({4}, input.data());
    auto tensorDst = createFloat32Tensor({4}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::absKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, SignKernelBasic) {
    std::vector<Float32> input = {-5.0f, 0.0f, 3.0f};
    std::vector<Float32> expected = {-1.0f, 0.0f, 1.0f};
    
    auto tensorSrc = createFloat32Tensor({3}, input.data());
    auto tensorDst = createFloat32Tensor({3}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::signKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, ExpKernelBasic) {
    std::vector<Float32> input = {0.0f, 1.0f, 2.0f};
    std::vector<Float32> expected = {1.0f, std::exp(1.0f), std::exp(2.0f)};
    
    auto tensorSrc = createFloat32Tensor({3}, input.data());
    auto tensorDst = createFloat32Tensor({3}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::expKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i], 1e-4f));
    }
}

TEST_F(CpuKernelTest, LogKernelBasic) {
    std::vector<Float32> input = {1.0f, std::exp(1.0f), std::exp(2.0f)};
    std::vector<Float32> expected = {0.0f, 1.0f, 2.0f};
    
    auto tensorSrc = createFloat32Tensor({3}, input.data());
    auto tensorDst = createFloat32Tensor({3}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::logKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i], 1e-4f));
    }
}

TEST_F(CpuKernelTest, SinKernelBasic) {
    std::vector<Float32> input = {0.0f, static_cast<Float32>(1.57079632679), static_cast<Float32>(3.14159265359)};
    std::vector<Float32> expected = {0.0f, 1.0f, 0.0f};
    
    auto tensorSrc = createFloat32Tensor({3}, input.data());
    auto tensorDst = createFloat32Tensor({3}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::sinKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i], 1e-5f));
    }
}

TEST_F(CpuKernelTest, CosKernelBasic) {
    std::vector<Float32> input = {0.0f, static_cast<Float32>(1.57079632679), static_cast<Float32>(3.14159265359)};
    std::vector<Float32> expected = {1.0f, 0.0f, -1.0f};
    
    auto tensorSrc = createFloat32Tensor({3}, input.data());
    auto tensorDst = createFloat32Tensor({3}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::cosKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i], 1e-5f));
    }
}

TEST_F(CpuKernelTest, CeilKernelBasic) {
    std::vector<Float32> input = {1.1f, -1.1f, 2.5f, -2.5f};
    std::vector<Float32> expected = {2.0f, -1.0f, 3.0f, -2.0f};
    
    auto tensorSrc = createFloat32Tensor({4}, input.data());
    auto tensorDst = createFloat32Tensor({4}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::ceilKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

TEST_F(CpuKernelTest, FloorKernelBasic) {
    std::vector<Float32> input = {1.1f, -1.1f, 2.5f, -2.5f};
    std::vector<Float32> expected = {1.0f, -2.0f, 2.0f, -3.0f};
    
    auto tensorSrc = createFloat32Tensor({4}, input.data());
    auto tensorDst = createFloat32Tensor({4}, input.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::floorKernel(tensorSrc, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(result[i], expected[i]));
    }
}

// ============== Larger Tensor Tests ==============

TEST_F(CpuKernelTest, AddKernelLargeTensor) {
    constexpr size_t size = 1000;
    std::vector<Float32> a(size);
    std::vector<Float32> b(size);
    
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<Float32>(i);
        b[i] = static_cast<Float32>(size - i);
    }
    
    auto tensorA = createFloat32Tensor({static_cast<Int64>(size)}, a.data());
    auto tensorB = createFloat32Tensor({static_cast<Int64>(size)}, b.data());
    auto tensorDst = createFloat32Tensor({static_cast<Int64>(size)}, a.data());
    
    ComputeContext ctx(backend::Device(0, backend::DeviceType::CPU), DataType::Float32);
    
    cpu::addKernel(tensorA, tensorB, tensorDst, ctx);
    
    Float32* result = tensorDst.storageRef().data().as<Float32>();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_TRUE(compareFloats(result[i], static_cast<Float32>(size)));
    }
}

} // namespace h3::core::compute
