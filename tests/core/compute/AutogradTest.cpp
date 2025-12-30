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

#include "Tensor.h"
#include <gtest/gtest.h>

using hahaha::Tensor;
using hahaha::math::NestedData;

class AutogradTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AutogradTest, SimpleAddition) {
    Tensor<float> a(10.0f);
    Tensor<float> b(20.0f);
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a + b;
    EXPECT_FLOAT_EQ(c.at({}), 30.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({}), 1.0f);
}

TEST_F(AutogradTest, SimpleMultiplication) {
    Tensor<float> x(3.0f);
    Tensor<float> y(4.0f);
    x.setRequiresGrad(true);
    y.setRequiresGrad(true);

    auto z = x * y;
    EXPECT_FLOAT_EQ(z.at({}), 12.0f);

    z.backward();

    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);
    EXPECT_FLOAT_EQ(x.grad()->at({}), 4.0f); // dz/dx = y = 4
    EXPECT_FLOAT_EQ(y.grad()->at({}), 3.0f); // dz/dy = x = 3
}

TEST_F(AutogradTest, ChainRule) {
    // z = x * y + b
    Tensor<float> x(2.0f);
    Tensor<float> y(3.0f);
    Tensor<float> b(5.0f);
    x.setRequiresGrad(true);
    y.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto mul_res = x * y;
    auto z = mul_res + b;

    EXPECT_FLOAT_EQ(z.at({}), 11.0f);

    z.backward();

    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    EXPECT_FLOAT_EQ(x.grad()->at({}), 3.0f); // dz/dx = y = 3
    EXPECT_FLOAT_EQ(y.grad()->at({}), 2.0f); // dz/dy = x = 2
    EXPECT_FLOAT_EQ(b.grad()->at({}), 1.0f); // dz/db = 1
}

TEST_F(AutogradTest, NodeReuse) {
    // y = x * x
    Tensor<float> x(5.0f);
    x.setRequiresGrad(true);

    auto y = x * x;
    EXPECT_FLOAT_EQ(y.at({}), 25.0f);

    y.backward();

    ASSERT_NE(x.grad(), nullptr);
    // dy/dx = x + x = 10
    EXPECT_FLOAT_EQ(x.grad()->at({}), 10.0f);
}

TEST_F(AutogradTest, MatrixMultiplication) {
    // C = A @ B
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    Tensor<float> A(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> B(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    A.setRequiresGrad(true);
    B.setRequiresGrad(true);

    auto C = A.matmul(B);
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //   = [[19, 22], [43, 50]]
    EXPECT_FLOAT_EQ(C.at({0, 0}), 19.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 50.0f);

    C.backward();

    ASSERT_NE(A.grad(), nullptr);
    ASSERT_NE(B.grad(), nullptr);

    // dL/dA = dL/dC @ B^T
    // If dL/dC = [[1, 1], [1, 1]] (since C.backward() starts with 1s)
    // B^T = [[5, 7], [6, 8]]
    // dL/dA = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]]
    //       = [[5+6, 7+8], [5+6, 7+8]]
    //       = [[11, 15], [11, 15]]
    EXPECT_FLOAT_EQ(A.grad()->at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(A.grad()->at({0, 1}), 15.0f);
    EXPECT_FLOAT_EQ(A.grad()->at({1, 0}), 11.0f);
    EXPECT_FLOAT_EQ(A.grad()->at({1, 1}), 15.0f);

    // dL/dB = A^T @ dL/dC
    // A^T = [[1, 3], [2, 4]]
    // dL/dB = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]]
    //       = [[1+3, 1+3], [2+4, 2+4]]
    //       = [[4, 4], [6, 6]]
    EXPECT_FLOAT_EQ(B.grad()->at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(B.grad()->at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(B.grad()->at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(B.grad()->at({1, 1}), 6.0f);
}

TEST_F(AutogradTest, SimpleSubtraction) {
    Tensor<float> a(30.0f);
    Tensor<float> b(10.0f);
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a - b;
    EXPECT_FLOAT_EQ(c.at({}), 20.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({}), 1.0f);  // dc/da = 1
    EXPECT_FLOAT_EQ(b.grad()->at({}), -1.0f); // dc/db = -1
}

TEST_F(AutogradTest, SimpleDivision) {
    Tensor<float> x(10.0f);
    Tensor<float> y(2.0f);
    x.setRequiresGrad(true);
    y.setRequiresGrad(true);

    auto z = x / y;
    EXPECT_FLOAT_EQ(z.at({}), 5.0f);

    z.backward();

    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);
    EXPECT_FLOAT_EQ(x.grad()->at({}), 0.5f);   // dz/dx = 1/y = 1/2 = 0.5
    EXPECT_FLOAT_EQ(y.grad()->at({}), -2.5f);  // dz/dy = -x/(y^2) = -10/4 = -2.5
}

