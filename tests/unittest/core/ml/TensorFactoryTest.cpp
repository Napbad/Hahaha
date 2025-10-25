#include <gtest/gtest.h>

#include "include/ml/common/TensorVar.h"

using namespace hahaha;
using core::ds::Vector;

// Tests for tensor<T>(...) helper factory returning TensorVarPtr

TEST(TensorFactoryTest, CreateScalar)
{
    auto s = tensor<float>(1.5f, "s");
    ASSERT_TRUE(s);

    const ml::TensorData<float>& td = s->tensorData();
    EXPECT_TRUE(td.isScalar());
    EXPECT_EQ(td.size(), static_cast<sizeT>(1));
    EXPECT_FLOAT_EQ(td[0], 1.5f);
}

TEST(TensorFactoryTest, CreateWithShape)
{
    auto v = tensor<float>(std::initializer_list<sizeT>{2}, "v");
    ASSERT_TRUE(v);
    EXPECT_EQ(v->shape(), Vector<sizeT>({2}));
    EXPECT_EQ(v->size(), static_cast<sizeT>(2));
    // default-initialized to 0
    EXPECT_FLOAT_EQ(v->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(v->data()[1], 0.0f);
}

TEST(TensorFactoryTest, CreateWithShape_NoName)
{
    // brace-init with integral elements (int) should be accepted
    auto v = tensor<float>({3});
    ASSERT_TRUE(v);
    EXPECT_EQ(v->shape(), Vector<sizeT>({3}));
    EXPECT_EQ(v->size(), static_cast<sizeT>(3));
}

TEST(TensorFactoryTest, CreateWithShapeAndData)
{
    auto m = tensor<float>(std::initializer_list<sizeT>{2, 2},
                           std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f},
                           "m");
    ASSERT_TRUE(m);
    EXPECT_EQ(m->shape(), Vector<sizeT>({2, 2}));
    EXPECT_EQ(m->size(), static_cast<sizeT>(4));
    EXPECT_FLOAT_EQ(m->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(m->data()[1], 2.0f);
    EXPECT_FLOAT_EQ(m->data()[2], 3.0f);
    EXPECT_FLOAT_EQ(m->data()[3], 4.0f);
}

TEST(TensorFactoryTest, CreateWithShapeAndData_NoName)
{
    auto m = tensor<float>({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    ASSERT_TRUE(m);
    EXPECT_EQ(m->shape(), Vector<sizeT>({2, 2}));
    EXPECT_EQ(m->size(), static_cast<sizeT>(4));
    EXPECT_FLOAT_EQ(m->data()[0], 10.0f);
    EXPECT_FLOAT_EQ(m->data()[1], 20.0f);
    EXPECT_FLOAT_EQ(m->data()[2], 30.0f);
    EXPECT_FLOAT_EQ(m->data()[3], 40.0f);
}

TEST(TensorFactoryTest, CreateScalar_NoName)
{
    auto s = tensor<float>(2.0f);
    ASSERT_TRUE(s);
    EXPECT_TRUE(s->isScalar());
    EXPECT_EQ(s->size(), static_cast<sizeT>(1));
    EXPECT_FLOAT_EQ(s->data()[0], 2.0f);
}


