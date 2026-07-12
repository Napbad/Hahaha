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
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
//
// Created by Cursor AI on 7/12/26.
//

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include "math/Scalar.h"
#include "backend/Device.h"

namespace h3::core::math {

class ScalarConvertTest : public ::testing::Test {
protected:
    backend::Device cpuDevice = backend::Device(0, backend::DeviceType::CPU);
};

// Test: Same type conversion (no-op)
TEST_F(ScalarConvertTest, SameTypeNoOp) {
    auto s = Scalar::zeros(DataType::Float32, cpuDevice);
    *s.as<Float32>() = 3.14f;
    s.convertToType(DataType::Float32);
    EXPECT_EQ(s.dtype(), DataType::Float32);
    EXPECT_NEAR(*s.as<Float32>(), 3.14f, 1e-6f);
}

// Test: Float32 to Float64
TEST_F(ScalarConvertTest, Float32ToFloat64) {
    auto s = Scalar::zeros(DataType::Float32, cpuDevice);
    *s.as<Float32>() = 3.14f;
    s.convertToType(DataType::Float64);
    EXPECT_EQ(s.dtype(), DataType::Float64);
    EXPECT_NEAR(*s.as<Float64>(), 3.14, 1e-6);
}

// Test: Float64 to Float32
TEST_F(ScalarConvertTest, Float64ToFloat32) {
    auto s = Scalar::zeros(DataType::Float64, cpuDevice);
    *s.as<Float64>() = 3.14159265358979;
    s.convertToType(DataType::Float32);
    EXPECT_EQ(s.dtype(), DataType::Float32);
    EXPECT_NEAR(*s.as<Float32>(), 3.14159f, 1e-5f);
}

// Test: Int32 to Int64
TEST_F(ScalarConvertTest, Int32ToInt64) {
    auto s = Scalar::zeros(DataType::Int32, cpuDevice);
    *s.as<Int32>() = 42;
    s.convertToType(DataType::Int64);
    EXPECT_EQ(s.dtype(), DataType::Int64);
    EXPECT_EQ(*s.as<Int64>(), 42);
}

// Test: Int64 to Int32 (truncation)
TEST_F(ScalarConvertTest, Int64ToInt32) {
    auto s = Scalar::zeros(DataType::Int64, cpuDevice);
    *s.as<Int64>() = 123456789;
    s.convertToType(DataType::Int32);
    EXPECT_EQ(s.dtype(), DataType::Int32);
    EXPECT_EQ(*s.as<Int32>(), 123456789);
}

// Test: Int32 to Float32
TEST_F(ScalarConvertTest, Int32ToFloat32) {
    auto s = Scalar::zeros(DataType::Int32, cpuDevice);
    *s.as<Int32>() = 42;
    s.convertToType(DataType::Float32);
    EXPECT_EQ(s.dtype(), DataType::Float32);
    EXPECT_NEAR(*s.as<Float32>(), 42.0f, 1e-6f);
}

// Test: Float32 to Int32 (rounding)
TEST_F(ScalarConvertTest, Float32ToInt32) {
    auto s = Scalar::zeros(DataType::Float32, cpuDevice);
    *s.as<Float32>() = 3.7f;
    s.convertToType(DataType::Int32);
    EXPECT_EQ(s.dtype(), DataType::Int32);
    EXPECT_EQ(*s.as<Int32>(), 4); // Should round to nearest
}

// Test: Float32 to Int32 (negative rounding)
TEST_F(ScalarConvertTest, Float32ToInt32Negative) {
    auto s = Scalar::zeros(DataType::Float32, cpuDevice);
    *s.as<Float32>() = -3.7f;
    s.convertToType(DataType::Int32);
    EXPECT_EQ(s.dtype(), DataType::Int32);
    EXPECT_EQ(*s.as<Int32>(), -4); // Should round to nearest
}

// Test: Unsigned to Signed
TEST_F(ScalarConvertTest, UInt32ToInt64) {
    auto s = Scalar::zeros(DataType::UInt32, cpuDevice);
    *s.as<UInt32>() = 100;
    s.convertToType(DataType::Int64);
    EXPECT_EQ(s.dtype(), DataType::Int64);
    EXPECT_EQ(*s.as<Int64>(), 100);
}

// Test: Signed to Unsigned
TEST_F(ScalarConvertTest, Int32ToUInt32) {
    auto s = Scalar::zeros(DataType::Int32, cpuDevice);
    *s.as<Int32>() = 42;
    s.convertToType(DataType::UInt32);
    EXPECT_EQ(s.dtype(), DataType::UInt32);
    EXPECT_EQ(*s.as<UInt32>(), 42u);
}

// Test: Int to Float
TEST_F(ScalarConvertTest, Int64ToFloat64) {
    auto s = Scalar::zeros(DataType::Int64, cpuDevice);
    *s.as<Int64>() = 12345678901234;
    s.convertToType(DataType::Float64);
    EXPECT_EQ(s.dtype(), DataType::Float64);
    EXPECT_NEAR(*s.as<Float64>(), 12345678901234.0, 1.0);
}

// Test: Float to Int
TEST_F(ScalarConvertTest, Float64ToInt64) {
    auto s = Scalar::zeros(DataType::Float64, cpuDevice);
    *s.as<Float64>() = 12345678.9;
    s.convertToType(DataType::Int64);
    EXPECT_EQ(s.dtype(), DataType::Int64);
    EXPECT_EQ(*s.as<Int64>(), 12345679);
}

// Test: Int8 conversion
TEST_F(ScalarConvertTest, Int32ToInt8) {
    auto s = Scalar::zeros(DataType::Int32, cpuDevice);
    *s.as<Int32>() = 127;
    s.convertToType(DataType::Int8);
    EXPECT_EQ(s.dtype(), DataType::Int8);
    EXPECT_EQ(*s.as<Int8>(), 127);
}

// Test: Int32 to UInt8
TEST_F(ScalarConvertTest, Int32ToUInt8) {
    auto s = Scalar::zeros(DataType::Int32, cpuDevice);
    *s.as<Int32>() = 200;
    s.convertToType(DataType::UInt8);
    EXPECT_EQ(s.dtype(), DataType::UInt8);
    EXPECT_EQ(*s.as<UInt8>(), 200);
}

// Test: UInt8 to Int32
TEST_F(ScalarConvertTest, UInt8ToInt32) {
    auto s = Scalar::zeros(DataType::UInt8, cpuDevice);
    *s.as<UInt8>() = 255;
    s.convertToType(DataType::Int32);
    EXPECT_EQ(s.dtype(), DataType::Int32);
    EXPECT_EQ(*s.as<Int32>(), 255);
}

// Test: Zero value conversions
TEST_F(ScalarConvertTest, ZeroConversions) {
    for (auto dtype : {DataType::Float32, DataType::Float64, DataType::Int32,
                        DataType::Int64, DataType::UInt32, DataType::UInt64}) {
        auto s = Scalar::zeros(dtype, cpuDevice);
        s.convertToType(DataType::Float64);
        EXPECT_EQ(s.dtype(), DataType::Float64);
        EXPECT_EQ(*s.as<Float64>(), 0.0);
    }
}

// Test: Large value conversions
TEST_F(ScalarConvertTest, LargeValueConversions) {
    auto s = Scalar::zeros(DataType::Int64, cpuDevice);
    *s.as<Int64>() = 9223372036854775807LL;
    s.convertToType(DataType::Float64);
    EXPECT_EQ(s.dtype(), DataType::Float64);
    // Note: large int64 may not be exactly representable in float64
    EXPECT_GT(*s.as<Float64>(), 9e18);
}

} // namespace h3::core::math
