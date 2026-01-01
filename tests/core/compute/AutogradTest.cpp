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
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#include <gtest/gtest.h>

#include "Tensor.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

class AutogradTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
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
    EXPECT_FLOAT_EQ(x.grad()->at({}), 0.5f);  // dz/dx = 1/y = 1/2 = 0.5
    EXPECT_FLOAT_EQ(y.grad()->at({}), -2.5f); // dz/dy = -x/(y^2) = -10/4 = -2.5
}

TEST_F(AutogradTest, TwoDim_Addition) {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> b(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a + b;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 12.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({1, 1}), 1.0f);
}

TEST_F(AutogradTest, TwoDim_Multiplication) {
    Tensor<float> x(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> y(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    x.setRequiresGrad(true);
    y.setRequiresGrad(true);

    auto z = x * y;
    EXPECT_FLOAT_EQ(z.at({0, 0}), 5.0f);  // 1*5
    EXPECT_FLOAT_EQ(z.at({0, 1}), 12.0f); // 2*6
    EXPECT_FLOAT_EQ(z.at({1, 0}), 21.0f); // 3*7
    EXPECT_FLOAT_EQ(z.at({1, 1}), 32.0f); // 4*8

    z.backward();

    ASSERT_NE(x.grad(), nullptr);
    ASSERT_NE(y.grad(), nullptr);
    EXPECT_FLOAT_EQ(x.grad()->at({0, 0}), 5.0f); // dz/dx = y
    EXPECT_FLOAT_EQ(x.grad()->at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(x.grad()->at({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(x.grad()->at({1, 1}), 8.0f);
    EXPECT_FLOAT_EQ(y.grad()->at({0, 0}), 1.0f); // dz/dy = x
    EXPECT_FLOAT_EQ(y.grad()->at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(y.grad()->at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(y.grad()->at({1, 1}), 4.0f);
}

TEST_F(AutogradTest, TwoDim_ChainRule) {
    // d = a * b + c
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> b(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    Tensor<float> c(NestedData<float>{{0.1f, 0.2f}, {0.3f, 0.4f}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);
    c.setRequiresGrad(true);

    auto ab = a * b;
    auto d = ab + c;

    EXPECT_FLOAT_EQ(d.at({0, 0}), 5.1f);  // 1*5 + 0.1
    EXPECT_FLOAT_EQ(d.at({1, 1}), 32.4f); // 4*8 + 0.4

    d.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);

    // d(d)/da = b (element-wise)
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 8.0f);

    // d(d)/db = a (element-wise)
    EXPECT_FLOAT_EQ(b.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({1, 1}), 4.0f);

    // d(d)/dc = 1 (element-wise)
    EXPECT_FLOAT_EQ(c.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.grad()->at({1, 1}), 1.0f);
}

TEST_F(AutogradTest, TwoDim_AddScalar) {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    a.setRequiresGrad(true);
    float scalar = 10.0f;

    auto c = a + scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 14.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 1.0f);
}

TEST_F(AutogradTest, TwoDim_SubtractScalar) {
    Tensor<float> a(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    a.setRequiresGrad(true);
    float scalar = 5.0f;

    auto c = a - scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 35.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 1.0f);
}

TEST_F(AutogradTest, TwoDim_MultiplyScalar) {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    a.setRequiresGrad(true);
    float scalar = 3.0f;

    auto c = a * scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 12.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 3.0f);
}

TEST_F(AutogradTest, TwoDim_DivideScalar) {
    Tensor<float> a(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    a.setRequiresGrad(true);
    float scalar = 2.0f;

    auto c = a / scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 20.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 0.5f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 0.5f);
}

TEST_F(AutogradTest, TwoDim_ScalarSubtractTensor) {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    a.setRequiresGrad(true);
    float scalar = 10.0f;

    auto c = scalar - a;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 6.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), -1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), -1.0f);
}

TEST_F(AutogradTest, TwoDim_ScalarDivideTensor) {
    Tensor<float> a(NestedData<float>{{2.0f, 4.0f}, {5.0f, 10.0f}});
    a.setRequiresGrad(true);
    float scalar = 20.0f;

    auto c = scalar / a;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 2.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    // d(scalar/a)/da = -scalar/(a^2)
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), -20.0f / (2.0f * 2.0f));   // -5.0f
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), -20.0f / (10.0f * 10.0f)); // -0.2f
}

TEST_F(AutogradTest, ThreeDim_Addition) {
    Tensor<float> a(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                      {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    Tensor<float> b(NestedData<float>{{{0.1f, 0.2f}, {0.3f, 0.4f}},
                                      {{0.5f, 0.6f}, {0.7f, 0.8f}}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a + b;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 1.1f); // 1.0 + 0.1
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 8.8f); // 8.0 + 0.8

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad()->at({0, 0, 0}), 1.0f);
}

TEST_F(AutogradTest, ThreeDim_Multiplication) {
    Tensor<float> a(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                      {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    Tensor<float> b(NestedData<float>{{{0.1f, 0.2f}, {0.3f, 0.4f}},
                                      {{0.5f, 0.6f}, {0.7f, 0.8f}}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a * b;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 0.1f); // 1.0 * 0.1
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 6.4f); // 8.0 * 0.8

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 0.1f); // dc/da = b
    EXPECT_FLOAT_EQ(b.grad()->at({0, 0, 0}), 1.0f); // dc/db = a
}

TEST_F(AutogradTest, ThreeDim_ComplexChain) {
    // f = (a * 2).reshape({4, 2}).transpose() + c
    Tensor<float> a(
        NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                          {{5.0f, 6.0f}, {7.0f, 8.0f}}}); // Shape {2, 2, 2}
    Tensor<float> c(NestedData<float>{
        {0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}}); // Shape {2, 4}
    a.setRequiresGrad(true);
    c.setRequiresGrad(true);

    auto a_times_2 = a * 2.0f;                   // Shape {2, 2, 2}
    auto a_reshaped = a_times_2.reshape({4, 2}); // Shape {4, 2}
    auto a_transposed = a_reshaped.transpose();  // Shape {2, 4}
    auto f = a_transposed + c;                   // Shape {2, 4}

    // Expected values for a_transposed:
    // a_times_2 = {{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}} (flattened:
    // 2,4,6,8,10,12,14,16) a_reshaped = {{2,4}, {6,8}, {10,12}, {14,16}}
    // a_transposed = {{2,6,10,14}, {4,8,12,16}}

    EXPECT_FLOAT_EQ(a_transposed.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a_transposed.at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(a_transposed.at({1, 3}), 16.0f);

    // Expected values for f:
    // f = {{2.1, 6.2, 10.3, 14.4}, {4.5, 8.6, 12.7, 16.8}}
    EXPECT_FLOAT_EQ(f.at({0, 0}), 2.1f);
    EXPECT_FLOAT_EQ(f.at({1, 3}), 16.8f);

    f.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);

    // d(f)/d(c) = 1 (element-wise), shape {2, 4}
    EXPECT_FLOAT_EQ(c.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.grad()->at({1, 3}), 1.0f);

    // d(f)/d(a) chain: d(f)/d(a_transposed) * d(a_transposed)/d(a_reshaped) *
    // d(a_reshaped)/d(a_times_2) * d(a_times_2)/d(a) d(f)/d(a_transposed) is
    // all 1s (from addition) -> shape {2, 4} d(a_transposed)/d(a_reshaped) is
    // transpose operation, so essentially it's d(a_reshaped).transpose()
    // d(a_reshaped)/d(a_times_2) is reshape, which just rearranges elements.
    // Gradients are preserved. d(a_times_2)/d(a) is multiplying by 2, so the
    // gradient is 2.

    // So d(f)/d(a) should be a.grad() should be 2.0f everywhere, reshaped and
    // transposed. Result should be 2.0f for each original element of 'a'.
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), 2.0f);
}

TEST_F(AutogradTest, ThreeDim_AddScalar) {
    Tensor<float> a(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                      {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    a.setRequiresGrad(true);
    float scalar = 10.0f;

    auto c = a + scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 18.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), 1.0f);
}

TEST_F(AutogradTest, ThreeDim_SubtractScalar) {
    Tensor<float> a(NestedData<float>{{{10.0f, 20.0f}, {30.0f, 40.0f}},
                                      {{50.0f, 60.0f}, {70.0f, 80.0f}}});
    a.setRequiresGrad(true);
    float scalar = 5.0f;

    auto c = a - scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 75.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), 1.0f);
}

TEST_F(AutogradTest, ThreeDim_MultiplyScalar) {
    Tensor<float> a(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                      {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    a.setRequiresGrad(true);
    float scalar = 3.0f;

    auto c = a * scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 24.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), 3.0f);
}

TEST_F(AutogradTest, ThreeDim_DivideScalar) {
    Tensor<float> a(NestedData<float>{{{10.0f, 20.0f}, {30.0f, 40.0f}},
                                      {{50.0f, 60.0f}, {70.0f, 80.0f}}});
    a.setRequiresGrad(true);
    float scalar = 2.0f;

    auto c = a / scalar;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 40.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 0.5f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), 0.5f);
}

TEST_F(AutogradTest, ThreeDim_ScalarSubtractTensor) {
    Tensor<float> a(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                      {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    a.setRequiresGrad(true);
    float scalar = 10.0f;

    auto c = scalar - a;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 2.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), -1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), -1.0f);
}

TEST_F(AutogradTest, ThreeDim_ScalarDivideTensor) {
    Tensor<float> a(NestedData<float>{{{2.0f, 4.0f}, {5.0f, 10.0f}},
                                      {{1.0f, 2.0f}, {4.0f, 5.0f}}});
    a.setRequiresGrad(true);
    float scalar = 20.0f;

    auto c = scalar / a;
    EXPECT_FLOAT_EQ(c.at({0, 0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1, 1}), 4.0f);

    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    // d(scalar/a)/da = -scalar/(a^2)
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), -20.0f / (2.0f * 2.0f)); // -5.0f
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1, 1}), -20.0f / (5.0f * 5.0f)); // -0.8f
}
