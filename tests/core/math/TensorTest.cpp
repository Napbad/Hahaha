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