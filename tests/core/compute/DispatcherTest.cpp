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

/// Tests for Dispatcher - the kernel registry and dispatch system

#include <gtest/gtest.h>
#include <vector>

#include "compute/Dispatcher.h"
#include "compute/kernels/CpuKernels.h"
#include "backend/Device.h"
#include "defines.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

class DispatcherTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    math::TensorInner createTensor(DataType dtype, const std::vector<Int64>& shape) {
        math::TensorMetadata meta;
        meta.dataType = dtype;
        meta.isContiguous = true;
        meta.isView = false;
        return math::TensorInner(shape, meta);
    }
};

TEST_F(DispatcherTest, SingletonInstance) {
    Dispatcher& instance1 = Dispatcher::instance();
    Dispatcher& instance2 = Dispatcher::instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(DispatcherTest, RegisterAndResolveUnary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Sqrt, DataType::Float32, backend::DeviceType::CPU);
    
    EXPECT_TRUE(disp.hasKernel(key, 1));
}

TEST_F(DispatcherTest, RegisterAndResolveBinary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    
    EXPECT_TRUE(disp.hasKernel(key, 2));
}

TEST_F(DispatcherTest, RegisterAndResolveTernary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Clamp, DataType::Float32, backend::DeviceType::CPU);
    
    EXPECT_TRUE(disp.hasKernel(key, 3));
}

TEST_F(DispatcherTest, AllDataTypesRegistered) {
    Dispatcher& disp = Dispatcher::instance();
    
    for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
        DispatchKey key(Operator::Add, static_cast<DataType>(dtype), backend::DeviceType::CPU);
        EXPECT_TRUE(disp.hasKernel(key, 2)) 
            << "Add kernel not registered for dtype " << dtype;
    }
}

TEST_F(DispatcherTest, UnregisteredKernelReturnsFalse) {
    Dispatcher& disp = Dispatcher::instance();
    // Use a very high operator value that shouldn't exist
    DispatchKey key(static_cast<Operator>(100), DataType::Float32, backend::DeviceType::CPU);
    
    EXPECT_FALSE(disp.hasKernel(key, 1));
    EXPECT_FALSE(disp.hasKernel(key, 2));
}

TEST_F(DispatcherTest, DispatchResultUnary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Sqrt, DataType::Float32, backend::DeviceType::CPU);
    
    auto result = disp.resolve(key, 1);
    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.kind, DispatchResult::Kind::Unary);
}

TEST_F(DispatcherTest, DispatchResultBinary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    
    auto result = disp.resolve(key, 2);
    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.kind, DispatchResult::Kind::Binary);
}

TEST_F(DispatcherTest, DispatchResultTernary) {
    Dispatcher& disp = Dispatcher::instance();
    DispatchKey key(Operator::Clamp, DataType::Float32, backend::DeviceType::CPU);
    
    auto result = disp.resolve(key, 3);
    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.kind, DispatchResult::Kind::Ternary);
}

TEST_F(DispatcherTest, RegistryStatistics) {
    Dispatcher& disp = Dispatcher::instance();
    
    size_t unaryCount = disp.unaryCount();
    size_t binaryCount = disp.binaryCount();
    size_t ternaryCount = disp.ternaryCount();
    
    EXPECT_GT(unaryCount, 0) << "No unary kernels registered";
    EXPECT_GT(binaryCount, 0) << "No binary kernels registered";
    EXPECT_GT(ternaryCount, 0) << "No ternary kernels registered";
}

} // namespace h3::core::compute
