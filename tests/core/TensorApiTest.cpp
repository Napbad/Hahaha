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

#include <gtest/gtest.h>

#include "Tensor.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

class TensorApiTest : public ::testing::Test {};

TEST_F(TensorApiTest, BuildFromVector_Creates1DTensor) {
    auto t = Tensor<int>::buildFromVector({1, 2, 3});
    EXPECT_EQ(t.getShape().size(), 1);
    EXPECT_EQ(t.getShape()[0], 3);
    EXPECT_EQ(t.at({0}), 1);
    EXPECT_EQ(t.at({2}), 3);
}

TEST_F(TensorApiTest, Data_ReturnsSharedPtr) {
    Tensor<float> t(3.0f);
    auto data = t.data();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->getTotalSize(), 1);
    EXPECT_FLOAT_EQ(data->at({}), 3.0f);
}

TEST_F(TensorApiTest, Grad_WhenNoBackward_ReturnsNullptr) {
    Tensor<float> t(3.0f);
    EXPECT_EQ(t.grad(), nullptr);
}

TEST_F(TensorApiTest, Clear_ResetsUnderlyingData) {
    Tensor<float> t(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    t.clear();
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 0.0f);
}

TEST_F(TensorApiTest, To_Gpu_ThrowsRuntimeError) {
    Tensor<float> t(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_THROW(t.to(hahaha::backend::Device(hahaha::backend::DeviceType::GPU, 0)),
                 std::runtime_error);
}

TEST_F(TensorApiTest, GetTotalSize_MatchesUnderlyingWrapper) {
    Tensor<int> t(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(t.getTotalSize(), 4);
}

TEST_F(TensorApiTest, ClearGrad_NoGrad_NoThrow) {
    Tensor<float> t(1.0f);
    EXPECT_NO_THROW(t.clearGrad());
}

TEST_F(TensorApiTest, GetComputeNodeAndSetComputeNode_Works) {
    Tensor<float> a(1.0f);
    Tensor<float> b(2.0f);
    auto nodeA = a.getComputeNode();
    auto nodeB = b.getComputeNode();
    ASSERT_NE(nodeA, nullptr);
    ASSERT_NE(nodeB, nullptr);

    a.setComputeNode(nodeB);
    EXPECT_FLOAT_EQ(a.at({}), 2.0f);

    // Restore for cleanliness
    a.setComputeNode(nodeA);
    EXPECT_FLOAT_EQ(a.at({}), 1.0f);
}

TEST_F(TensorApiTest, RequiresGrad_FlagIsStoredOnNode) {
    Tensor<float> t(1.0f);
    EXPECT_FALSE(t.getRequiresGrad());
    t.setRequiresGrad(true);
    EXPECT_TRUE(t.getRequiresGrad());
}

TEST_F(TensorApiTest, Sum_ForwardsToTensorWrapper) {
    Tensor<int> t(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(t.sum(), 10);
}


