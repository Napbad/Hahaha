#include <gtest/gtest.h>

#include "core/compute/Variable.h"

using namespace hahaha::ml;
using hahaha::core::ds::Vector;

TEST(VariableOpTest, AddSubMulDivForward)
{
    auto a = Variable<float>(Tensor<float>({2}, {1.0f, 2.0f}), true);
    auto b = Variable<float>(Tensor<float>({2}, {3.0f, 4.0f}), true);

    auto c = a + b; // [4,6]
    auto d = a - b; // [-2,-2]
    auto e = a * b; // [3,8]
    auto f = b / a; // [3,2]

    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(c)[0], 4.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(c)[1], 6.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(d)[0], -2.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(d)[1], -2.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(e)[0], 3.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(e)[1], 8.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(f)[0], 3.0f);
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(f)[1], 2.0f);
}

TEST(VariableOpTest, MatmulForward)
{
    // a: 1x2, b: 2x1 => out: 1x1
    auto a = Variable<float>(Tensor<float>({1, 2}, {1.0f, 2.0f}), true);
    auto b = Variable<float>(Tensor<float>({2, 1}, {3.0f, 4.0f}), true);
    auto out = a.matmul(b);
    // out = [1*3 + 2*4] = [11]
    EXPECT_EQ(out.shape(), Vector<sizeT>({1, 1}));
    EXPECT_FLOAT_EQ(static_cast<Tensor<float>&>(out)[0], 11.0f);
}

TEST(VariableOpTest, ReLUBackwardMask)
{
    auto x = Variable<float>(Tensor<float>({4}, {-2.0f, -0.5f, 0.0f, 3.0f}), true);
    auto y = x.relu();
    y.backward();
    auto& gx = x.grad();
    EXPECT_FLOAT_EQ(gx[0], 0.0f);
    EXPECT_FLOAT_EQ(gx[1], 0.0f);
    EXPECT_FLOAT_EQ(gx[2], 0.0f);
    EXPECT_FLOAT_EQ(gx[3], 1.0f);
}

TEST(VariableOpTest, SigmoidBackwardNonZero)
{
    auto x = Variable<float>(Tensor<float>({2}, {0.0f, 2.0f}), true);
    auto y = x.sigmoid();
    y.backward();
    auto& gx = x.grad();
    EXPECT_GT(gx[0], 0.0f);
    EXPECT_GT(gx[1], 0.0f);
}

TEST(VariableOpTest, SumAndMeanBackward)
{
    auto x = Variable<float>(Tensor<float>({3}, {1.0f, 2.0f, 3.0f}), true);
    auto s = x.sum();
    s.backward();
    auto gx = x.grad();
    EXPECT_FLOAT_EQ(gx[0], 1.0f);
    EXPECT_FLOAT_EQ(gx[1], 1.0f);
    EXPECT_FLOAT_EQ(gx[2], 1.0f);

    x.zeroGrad();
    auto m = x.mean();
    m.backward();
    gx = x.grad();
    EXPECT_FLOAT_EQ(gx[0], 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(gx[1], 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(gx[2], 1.0f / 3.0f);
}

TEST(VariableOpTest, DataAccessorAndInplaceOps)
{
    auto a = Variable<float>(Tensor<float>({3}, {1.0f, 2.0f, 3.0f}), false);
    // data() accessor should expose underlying vector
    EXPECT_FLOAT_EQ(a.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(a.data()[2], 3.0f);

    // In-place compound ops on tensor
    Tensor<float> delta({3}, {0.5f, 0.5f, 0.5f});
    a.tensor() += delta; // 1.5, 2.5, 3.5
    EXPECT_FLOAT_EQ(a.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(a.data()[1], 2.5f);
    EXPECT_FLOAT_EQ(a.data()[2], 3.5f);

    // scalar variants
    a.tensor() = a.tensor() - 0.5f; // 1.0, 2.0, 3.0
    EXPECT_FLOAT_EQ(a.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(a.data()[2], 3.0f);
}

