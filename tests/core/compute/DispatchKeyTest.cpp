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

/// Tests for DispatchKey - the compressed dispatch key used for kernel lookup

#include <gtest/gtest.h>

#include "compute/DispatchKey.h"
#include "backend/Device.h"
#include "defines.h"

namespace h3::core::compute {

class DispatchKeyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DispatchKeyTest, DefaultConstruction) {
    DispatchKey key;
    EXPECT_EQ(key.op(), 0);
    EXPECT_EQ(key.dtype(), 0);
    EXPECT_EQ(key.device(), 0);
}

TEST_F(DispatchKeyTest, ConstructionWithEnums) {
    DispatchKey key(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    EXPECT_EQ(key.op(), static_cast<uint8_t>(Operator::Add));
    EXPECT_EQ(key.dtype(), static_cast<uint8_t>(DataType::Float32));
    EXPECT_EQ(key.device(), static_cast<uint8_t>(backend::DeviceType::CPU));
}

TEST_F(DispatchKeyTest, PackedRepresentation) {
    DispatchKey key(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    uint64_t packed = key.packed();
    
    // Expected: (op << 16) | (dtype << 8) | device
    uint64_t expected = (static_cast<uint64_t>(Operator::Add) << 16) | 
                        (static_cast<uint64_t>(DataType::Float32) << 8) | 
                        static_cast<uint64_t>(backend::DeviceType::CPU);
    EXPECT_EQ(packed, expected);
}

TEST_F(DispatchKeyTest, UnpackedFromPacked) {
    DispatchKey original(Operator::Mul, DataType::Float64, backend::DeviceType::CUDA);
    uint64_t packed = original.packed();
    
    DispatchKey restored = DispatchKey::fromPacked(packed);
    
    EXPECT_EQ(restored.opEnum(), Operator::Mul);
    EXPECT_EQ(restored.dtypeEnum(), DataType::Float64);
    EXPECT_EQ(restored.deviceEnum(), backend::DeviceType::CUDA);
}

TEST_F(DispatchKeyTest, Equality) {
    DispatchKey key1(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    DispatchKey key2(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    DispatchKey key3(Operator::Sub, DataType::Float32, backend::DeviceType::CPU);
    
    EXPECT_TRUE(key1 == key2);
    EXPECT_FALSE(key1 == key3);
    EXPECT_TRUE(key1 != key3);
}

TEST_F(DispatchKeyTest, DifferentOperators) {
    for (int op = 0; op < static_cast<int>(Operator::Count); ++op) {
        DispatchKey key(static_cast<Operator>(op), DataType::Float32, backend::DeviceType::CPU);
        EXPECT_EQ(key.opEnum(), static_cast<Operator>(op));
    }
}

TEST_F(DispatchKeyTest, DifferentDataTypes) {
    for (int dtype = 1; dtype < static_cast<int>(DataType::Count); ++dtype) {
        DispatchKey key(Operator::Add, static_cast<DataType>(dtype), backend::DeviceType::CPU);
        EXPECT_EQ(key.dtypeEnum(), static_cast<DataType>(dtype));
    }
}

TEST_F(DispatchKeyTest, DifferentDevices) {
    DispatchKey keyCpu(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    DispatchKey keyCuda(Operator::Add, DataType::Float32, backend::DeviceType::CUDA);
    
    EXPECT_EQ(keyCpu.deviceEnum(), backend::DeviceType::CPU);
    EXPECT_EQ(keyCuda.deviceEnum(), backend::DeviceType::CUDA);
    EXPECT_NE(keyCpu, keyCuda);
}

TEST_F(DispatchKeyTest, ToString) {
    DispatchKey key(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    std::string str = key.toString();
    
    // toString outputs integers, check for presence of numeric values
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("DispatchKey"), std::string::npos);
}

TEST_F(DispatchKeyTest, HashConsistency) {
    DispatchKey key1(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    DispatchKey key2(Operator::Add, DataType::Float32, backend::DeviceType::CPU);
    DispatchKey key3(Operator::Sub, DataType::Float32, backend::DeviceType::CPU);
    
    DispatchKeyHash hasher;
    EXPECT_EQ(hasher(key1), hasher(key2));
    EXPECT_NE(hasher(key1), hasher(key3));
}

} // namespace h3::core::compute
