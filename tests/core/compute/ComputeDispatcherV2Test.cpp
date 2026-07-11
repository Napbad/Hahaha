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

/// Tests for ComputeDispatcherV2 - the public-facing dispatch API

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "compute/Dispatcher.h"
#include "backend/Device.h"
#include "defines.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class ComputeDispatcherV2Test : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    bool compareFloats(Float32 a, Float32 b, Float32 epsilon = 1e-5f) {
        return std::fabs(a - b) < epsilon;
    }
};

// Helper to create tensors for testing
math::TensorInner createTestTensor(const std::vector<Float32>& data) {
    math::TensorShape shape({static_cast<Int64>(data.size())});
    math::TensorMetadata meta;
    meta.dataType = DataType::Float32;
    meta.device = backend::Device(0, backend::DeviceType::CPU);
    meta.isContiguous = true;
    meta.isView = false;
    
    math::TensorInner tensor(shape, meta);
    
    if (!data.empty()) {
        Float32* tensorData = tensor.storageRef().data().as<Float32>();
        for (size_t i = 0; i < data.size(); ++i) {
            tensorData[i] = data[i];
        }
    }
    
    return tensor;
}

TEST_F(ComputeDispatcherV2Test, DispatchAdd) {
    std::vector<Float32> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> b = {5.0f, 4.0f, 3.0f, 2.0f};
    std::vector<Float32> expected = {6.0f, 6.0f, 6.0f, 6.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    // Operands format: [src0, src1, ..., dst] where last is output
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Add, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchSub) {
    std::vector<Float32> a = {10.0f, 8.0f, 6.0f, 4.0f};
    std::vector<Float32> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> expected = {9.0f, 6.0f, 3.0f, 0.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Sub, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchMul) {
    std::vector<Float32> a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<Float32> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<Float32> expected = {2.0f, 6.0f, 12.0f, 20.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Mul, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchDiv) {
    std::vector<Float32> a = {10.0f, 9.0f, 8.0f, 6.0f};
    std::vector<Float32> b = {2.0f, 3.0f, 4.0f, 2.0f};
    std::vector<Float32> expected = {5.0f, 3.0f, 2.0f, 3.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Div, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchSqrt) {
    std::vector<Float32> input = {1.0f, 4.0f, 9.0f, 16.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto tensorSrc = utils::make_own_ptr<math::TensorInner>(createTestTensor(input));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    // For unary: [src, dst]
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorSrc));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Sqrt, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[1]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchAbs) {
    std::vector<Float32> input = {-1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto tensorSrc = utils::make_own_ptr<math::TensorInner>(createTestTensor(input));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorSrc));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Abs, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[1]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchMax) {
    std::vector<Float32> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<Float32> b = {4.0f, 2.0f, 7.0f, 6.0f};
    std::vector<Float32> expected = {4.0f, 5.0f, 7.0f, 8.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Max, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchMin) {
    std::vector<Float32> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<Float32> b = {4.0f, 2.0f, 7.0f, 6.0f};
    std::vector<Float32> expected = {1.0f, 2.0f, 3.0f, 6.0f};
    
    auto tensorA = utils::make_own_ptr<math::TensorInner>(createTestTensor(a));
    auto tensorB = utils::make_own_ptr<math::TensorInner>(createTestTensor(b));
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorA));
    operands.push_back(std::move(tensorB));
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Min, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_TRUE(result.has_value());
    
    Float32* dstData = operands[2]->storageRef().data().as<Float32>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(compareFloats(dstData[i], expected[i]));
    }
}

TEST_F(ComputeDispatcherV2Test, DispatchEmptyOperands) {
    auto tensorDst = utils::make_own_ptr<math::TensorInner>(createTestTensor({}));
    
    std::vector<utils::OwnPointer<math::TensorInner>> operands;
    operands.push_back(std::move(tensorDst));
    
    auto result = Dispatcher::instance().dispatch(
        Operator::Add, backend::DeviceType::CPU, DataType::Float32, operands);
    EXPECT_FALSE(result.has_value());
}

} // namespace h3::core::compute
