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


#include "math/Tensor.h"

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
  void TearDown() override {

  }
};

using hahaha::math::Tensor;

TEST_F(TensorTest, BasicProperties) { 
    Tensor<int> t({{1, 2}, {3, 4}});
    EXPECT_EQ(t.shape().totalSize(), 4);
    EXPECT_EQ(t.shape().dims().size(), 2);
    EXPECT_EQ(t.shape().dims()[0], 2);
    EXPECT_EQ(t.shape().dims()[1], 2);
    
    EXPECT_EQ(t.stride().size(), 2);
    EXPECT_EQ(t.stride()[0], 2);
    EXPECT_EQ(t.stride()[1], 1);
}

TEST_F(TensorTest, MoveSemantics) {
    Tensor<int> t1({{1, 2}, {3, 4}});
    Tensor<int> t2(std::move(t1));
    
    EXPECT_EQ(t2.shape().totalSize(), 4);
    // t1's state is valid but unspecified, typically its data pointer is null
}

TEST_F(TensorTest, SingleValueTensor) {
    Tensor<float> t(42.0f);
    EXPECT_EQ(t.shape().totalSize(), 1);
    EXPECT_EQ(t.shape().dims().size(), 0);
}

TEST_F(TensorTest, HigherDimensionalTensor) {
    Tensor<int> t({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(t.shape().totalSize(), 8);
    EXPECT_EQ(t.shape().dims().size(), 3);
    EXPECT_EQ(t.shape().dims()[0], 2);
    EXPECT_EQ(t.shape().dims()[1], 2);
    EXPECT_EQ(t.shape().dims()[2], 2);
    
    EXPECT_EQ(t.stride()[0], 4);
    EXPECT_EQ(t.stride()[1], 2);
    EXPECT_EQ(t.stride()[2], 1);
}

TEST_F(TensorTest, ElementAccess) {
    Tensor<int> t({{1, 2}, {3, 4}});
    EXPECT_EQ(t.at({0, 0}), 1);
    EXPECT_EQ(t.at({0, 1}), 2);
    EXPECT_EQ(t.at({1, 0}), 3);
    EXPECT_EQ(t.at({1, 1}), 4);
    
    t.at({1, 1}) = 10;
    EXPECT_EQ(t.at({1, 1}), 10);
    
    EXPECT_THROW(t.at({0, 0, 0}), std::out_of_range);
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
}

TEST_F(TensorTest, ArithmeticOperations) {
    Tensor<float> t1({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> t2({{5.0f, 6.0f}, {7.0f, 8.0f}});
    
    auto t_add = t1 + t2;
    EXPECT_EQ(t_add.at({0, 0}), 6.0f);
    EXPECT_EQ(t_add.at({1, 1}), 12.0f);
    
    auto t_sub = t1 - t2;
    EXPECT_EQ(t_sub.at({0, 0}), -4.0f);
    
    auto t_mul = t1 * t2;
    EXPECT_EQ(t_mul.at({0, 1}), 12.0f);
    
    auto t_div = t2 / t1;
    EXPECT_EQ(t_div.at({1, 0}), 7.0f / 3.0f);
    
    auto t_neg = -t1;
    EXPECT_EQ(t_neg.at({0, 0}), -1.0f);
}

TEST_F(TensorTest, Reshape) {
    Tensor<int> t({1, 2, 3, 4, 5, 6});
    auto t2 = t.reshape({2, 3});
    EXPECT_EQ(t2.shape().dims().size(), 2);
    EXPECT_EQ(t2.at({0, 0}), 1);
    EXPECT_EQ(t2.at({1, 2}), 6);
    
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
}

TEST_F(TensorTest, MatrixMultiplication) {
    Tensor<float> A({{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    Tensor<float> B({{5.0f, 6.0f}, {7.0f, 8.0f}}); // 2x2
    
    auto C = A.matmul(B);
    // [1 2] [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4] [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    EXPECT_EQ(C.at({0, 0}), 19.0f);
    EXPECT_EQ(C.at({0, 1}), 22.0f);
    EXPECT_EQ(C.at({1, 0}), 43.0f);
    EXPECT_EQ(C.at({1, 1}), 50.0f);
}

TEST_F(TensorTest, Transpose) {
    Tensor<int> t({{1, 2, 3}, {4, 5, 6}}); // 2x3
    auto tt = t.transpose(); // 3x2
    
    EXPECT_EQ(tt.shape().dims()[0], 3);
    EXPECT_EQ(tt.shape().dims()[1], 2);
    EXPECT_EQ(tt.at({0, 0}), 1);
    EXPECT_EQ(tt.at({0, 1}), 4);
    EXPECT_EQ(tt.at({2, 1}), 6);
}