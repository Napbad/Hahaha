#include <gtest/gtest.h>

#include "include/ml/common/TensorVar.h"

using namespace hahaha;
using core::ds::Vector;

TEST(TensorVarTest, BasicsShapeSizeData)
{
    TensorVar a({2}, {1.0f, 2.0f});
    EXPECT_EQ(a.shape(), Vector<sizeT>({2}));
    EXPECT_EQ(a.size(), static_cast<sizeT>(2));
    EXPECT_FLOAT_EQ(a.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.data()[1], 2.0f);
}

TEST(TensorVarTest, PointerLikeForwardingTranspose)
{
    TensorVar<float> m({1, 2}, {1.0f, 2.0f});
    auto mt = m->transpose();
    EXPECT_EQ(mt.shape(), Vector<sizeT>({2, 1}));
    EXPECT_FLOAT_EQ(mt[0], 1.0f);
    EXPECT_FLOAT_EQ(mt[1], 2.0f);
}

TEST(TensorVarTest, DerefAndConversionToTensor)
{
    TensorVar<float> v({3}, {1.0f, 2.0f, 3.0f});
    ml::TensorData<float>& t = *v; // operator*()
    EXPECT_EQ(t.size(), static_cast<sizeT>(3));

    const auto& ct = static_cast<const ml::TensorData<float>&>(v);
    EXPECT_FLOAT_EQ(ct.sum(), 6.0f);
}

TEST(TensorVarTest, EmptyThrowsOnAccess)
{
    TensorVar<float> empty;
    EXPECT_THROW(empty.size(), std::runtime_error);
}

