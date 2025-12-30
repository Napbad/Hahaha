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

#include <cstdlib>
#include <gtest/gtest.h>

#include "math/TensorWrapper.h"

class TensorTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

using hahaha::math::TensorWrapper;

using hahaha::math::NestedData;

TEST_F(TensorTest, BasicProperties) {
    TensorWrapper<int> tensor_2d(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_2d.getShape().getTotalSize(), 4);
    EXPECT_EQ(tensor_2d.getShape().getDims().size(), 2);
    EXPECT_EQ(tensor_2d.getShape().getDims()[0], 2);
    EXPECT_EQ(tensor_2d.getShape().getDims()[1], 2);

    EXPECT_EQ(tensor_2d.getStride().getSize(), 2);
    EXPECT_EQ(tensor_2d.getStride()[0], 2);
    EXPECT_EQ(tensor_2d.getStride()[1], 1);
}

TEST_F(TensorTest, MoveSemantics) {
    TensorWrapper<int> tensor_orig(NestedData<int>{{1, 2}, {3, 4}});
    TensorWrapper<int> tensor_moved(std::move(tensor_orig));

    EXPECT_EQ(tensor_moved.getShape().getTotalSize(), 4);
    // tensor_orig's state is valid but unspecified
}

TEST_F(TensorTest, SingleValueTensor) {
    TensorWrapper<float> tensor_single(42.0f);
    EXPECT_EQ(tensor_single.getShape().getTotalSize(), 1);
    EXPECT_EQ(tensor_single.getShape().getDims().size(), 0);
}

TEST_F(TensorTest, HigherDimensionalTensor) {
    TensorWrapper<int> tensor_3d(
        NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(tensor_3d.getShape().getTotalSize(), 8);
    EXPECT_EQ(tensor_3d.getShape().getDims().size(), 3);
    EXPECT_EQ(tensor_3d.getShape().getDims()[0], 2);
    EXPECT_EQ(tensor_3d.getShape().getDims()[1], 2);
    EXPECT_EQ(tensor_3d.getShape().getDims()[2], 2);

    EXPECT_EQ(tensor_3d.getStride()[0], 4);
    EXPECT_EQ(tensor_3d.getStride()[1], 2);
    EXPECT_EQ(tensor_3d.getStride()[2], 1);
}

TEST_F(TensorTest, ElementAccess) {
    TensorWrapper<int> tensor_access(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_access.at({0, 0}), 1);
    EXPECT_EQ(tensor_access.at({0, 1}), 2);
    EXPECT_EQ(tensor_access.at({1, 0}), 3);
    EXPECT_EQ(tensor_access.at({1, 1}), 4);

    tensor_access.at({1, 1}) = 10;
    EXPECT_EQ(tensor_access.at({1, 1}), 10);

    EXPECT_THROW(tensor_access.at({0, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor_access.at({2, 0}), std::out_of_range);
}

TEST_F(TensorTest, ArithmeticEdgeCases) {
    TensorWrapper<float> tensor_a(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> tensor_b(NestedData<float>{{1.0f, 2.0f, 3.0f}});

    // Shape mismatch
    EXPECT_THROW(tensor_a + tensor_b, std::invalid_argument);
    EXPECT_THROW(tensor_a - tensor_b, std::invalid_argument);
    EXPECT_THROW(tensor_a * tensor_b, std::invalid_argument);
    EXPECT_THROW(tensor_a / tensor_b, std::invalid_argument);

    // Division by zero
    TensorWrapper<float> tensor_c(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> tensor_zero(NestedData<float>{{0.0f, 1.0f}});
    EXPECT_THROW(tensor_c / tensor_zero, std::runtime_error);
}

TEST_F(TensorTest, OperatorInPlaceAddition) {
    TensorWrapper<int> tensor1(NestedData<int>{{1, 2}, {3, 4}});
    TensorWrapper<int> tensor2(NestedData<int>{{5, 6}, {7, 8}});

    tensor1 += tensor2;
    EXPECT_EQ(tensor1.at({0, 0}), 6);
    EXPECT_EQ(tensor1.at({1, 1}), 12);

    TensorWrapper<int> tensor3(NestedData<int>{1, 2});
    EXPECT_THROW(tensor1 += tensor3, std::invalid_argument);
}

TEST_F(TensorTest, ReshapeEdgeCases) {
    TensorWrapper<int> tensor_reshape(NestedData<int>{1, 2, 3, 4});

    // Size mismatch
    EXPECT_THROW(tensor_reshape.reshape({3}), std::invalid_argument);
    EXPECT_THROW(tensor_reshape.reshape({2, 3}), std::invalid_argument);
}

TEST_F(TensorTest, MatmulEdgeCases) {
    TensorWrapper<float> matrix_a(
        NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});                   // 2x2
    TensorWrapper<float> matrix_b(NestedData<float>{{1.0f, 2.0f, 3.0f}}); // 1x3

    // Dimension mismatch (1D vs 2D)
    TensorWrapper<float> tensor_c(NestedData<float>{1.0f, 2.0f});
    EXPECT_THROW(matrix_a.matmul(tensor_c), std::invalid_argument);

    // Inner dimension mismatch
    TensorWrapper<float> matrix_d(
        NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}); // 3x2
    EXPECT_THROW(matrix_a.matmul(matrix_d), std::invalid_argument);
}

TEST_F(TensorTest, TransposeEdgeCases) {
    TensorWrapper<int> tensor_1d(NestedData<int>{1, 2, 3}); // 1D
    EXPECT_THROW(tensor_1d.transpose(), std::invalid_argument);
}

TEST_F(TensorTest, BroadcastUnimplemented) {
    TensorWrapper<int> tensor1(NestedData<int>{1});
    TensorWrapper<int> tensor2(NestedData<int>{1, 2});
    EXPECT_THROW(tensor1.broadcast(tensor2), std::runtime_error);
}

TEST_F(TensorTest, ArithmeticOperations) {
    TensorWrapper<float> tensor1(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> tensor2(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});

    auto tensor_add = tensor1 + tensor2;
    EXPECT_EQ(tensor_add.at({0, 0}), 6.0f);
    EXPECT_EQ(tensor_add.at({1, 1}), 12.0f);

    auto tensor_sub = tensor1 - tensor2;
    EXPECT_EQ(tensor_sub.at({0, 0}), -4.0f);

    auto tensor_mul = tensor1 * tensor2;
    EXPECT_EQ(tensor_mul.at({0, 1}), 12.0f);

    auto tensor_div = tensor2 / tensor1;
    EXPECT_EQ(tensor_div.at({1, 0}), 7.0f / 3.0f);

    auto tensor_neg = -tensor1;
    EXPECT_EQ(tensor_neg.at({0, 0}), -1.0f);
}

TEST_F(TensorTest, Reshape) {
    TensorWrapper<int> tensor_orig(NestedData<int>{1, 2, 3, 4, 5, 6});
    auto tensor_reshaped = tensor_orig.reshape({2, 3});
    EXPECT_EQ(tensor_reshaped.getShape().getDims().size(), 2);
    EXPECT_EQ(tensor_reshaped.at({0, 0}), 1);
    EXPECT_EQ(tensor_reshaped.at({1, 2}), 6);

    EXPECT_THROW(tensor_orig.reshape({2, 2}), std::invalid_argument);
}

TEST_F(TensorTest, MatrixMultiplication) {
    TensorWrapper<float> matrix_a(
        NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    TensorWrapper<float> matrix_b(
        NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}}); // 2x2

    auto matrix_c = matrix_a.matmul(matrix_b);
    EXPECT_EQ(matrix_c.at({0, 0}), 19.0f);
    EXPECT_EQ(matrix_c.at({0, 1}), 22.0f);
    EXPECT_EQ(matrix_c.at({1, 0}), 43.0f);
    EXPECT_EQ(matrix_c.at({1, 1}), 50.0f);
}

TEST_F(TensorTest, Transpose) {
    TensorWrapper<int> tensor_orig(
        NestedData<int>{{1, 2, 3}, {4, 5, 6}});       // 2x3
    auto tensor_transposed = tensor_orig.transpose(); // 3x2

    EXPECT_EQ(tensor_transposed.getShape().getDims()[0], 3);
    EXPECT_EQ(tensor_transposed.getShape().getDims()[1], 2);
    EXPECT_EQ(tensor_transposed.at({0, 0}), 1);
    EXPECT_EQ(tensor_transposed.at({0, 1}), 4);
    EXPECT_EQ(tensor_transposed.at({2, 1}), 6);
}
