// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
// 

#include <gtest/gtest.h>
#include "math/ds/TensorStride.h"
#include "math/ds/TensorShape.h"

using hahaha::math::TensorStride;
using hahaha::math::TensorShape;

class TensorStrideTest : public ::testing::Test {};

TEST_F(TensorStrideTest, StrideFromVector1D) {
    std::vector<uint32_t> dims = {5};
    TensorStride stride(dims);
    ASSERT_EQ(stride.size(), 1);
    ASSERT_EQ(stride[0], 1);
}

TEST_F(TensorStrideTest, StrideFromVector2D) {
    std::vector<uint32_t> dims = {3, 4};
    TensorStride stride(dims);
    ASSERT_EQ(stride.size(), 2);
    ASSERT_EQ(stride[0], 4);
    ASSERT_EQ(stride[1], 1);
}

TEST_F(TensorStrideTest, StrideFromVector3D) {
    std::vector<uint32_t> dims = {2, 3, 4};
    TensorStride stride(dims);
    ASSERT_EQ(stride.size(), 3);
    ASSERT_EQ(stride[0], 12);
    ASSERT_EQ(stride[1], 4);
    ASSERT_EQ(stride[2], 1);
}

TEST_F(TensorStrideTest, StrideFromShape) {
    TensorShape shape({2, 3, 4});
    TensorStride stride(shape);
    ASSERT_EQ(stride.size(), 3);
    ASSERT_EQ(stride[0], 12);
    ASSERT_EQ(stride[1], 4);
    ASSERT_EQ(stride[2], 1);
}

TEST_F(TensorStrideTest, EmptyStride) {
    TensorShape shape({});
    TensorStride stride(shape);
    ASSERT_EQ(stride.size(), 0);
}

TEST_F(TensorStrideTest, ToString) {
    TensorStride stride(TensorShape({2, 3}));
    ASSERT_EQ(stride.toString(), "[3, 1]");
}

TEST_F(TensorStrideTest, Reverse) {
    TensorStride stride(TensorShape({2, 3}));
    stride.reverse();
    ASSERT_EQ(stride[0], 1);
    ASSERT_EQ(stride[1], 3);
}