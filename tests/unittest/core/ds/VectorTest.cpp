#include <gtest/gtest.h>
#include <vector>

#include <ds/Vector.h>

using namespace hahaha::core;
using namespace hahaha::core::ds;

// Test fixture for Vector
template <typename T> class VectorTest : public ::testing::Test
{
  protected:
    // You can add shared setup/teardown logic here if needed
    void SetUp() override
    {
        // core setup for tests
    }

    void TearDown() override
    {
        // core teardown for tests
    }
};

// Define type-parameterized tests
using TestTypes = ::testing::Types<int, hahaha::f32, double>;
TYPED_TEST_SUITE(VectorTest, TestTypes);

TYPED_TEST(VectorTest, DefaultConstructor)
{
    Vector<TypeParam> vec;
    ASSERT_EQ(vec.size(), 0);
    ASSERT_EQ(vec.capacity(), 0);
    ASSERT_TRUE(vec.empty());
}

TYPED_TEST(VectorTest, CountConstructor)
{
    Vector<TypeParam> vec(5);
    ASSERT_EQ(vec.size(), 0);
    ASSERT_GE(vec.capacity(), 5);
    ASSERT_TRUE(vec.empty());
}

TYPED_TEST(VectorTest, InitializerListConstructor)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1),
                             static_cast<TypeParam>(2),
                             static_cast<TypeParam>(3),
                             static_cast<TypeParam>(4),
                             static_cast<TypeParam>(5)};
    ASSERT_EQ(vec.size(), 5);
    ASSERT_EQ(vec.capacity(), 5);
    for (size_t i = 0; i < 5; ++i)
    {
        ASSERT_EQ(vec[i], static_cast<TypeParam>(i + 1));
    }
}

TYPED_TEST(VectorTest, IteratorConstructor)
{
    Vector<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vector<TypeParam> vec(original.begin(), original.end());
    ASSERT_EQ(vec.size(), 3);
    ASSERT_EQ(vec.capacity(), 3);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(1));
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
    ASSERT_EQ(vec[2], static_cast<TypeParam>(3));
}

TYPED_TEST(VectorTest, CopyConstructor)
{
    Vector<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vector<TypeParam> copied = original;

    ASSERT_EQ(copied.size(), original.size());
    ASSERT_EQ(copied.capacity(), original.capacity());
    for (size_t i = 0; i < original.size(); ++i)
    {
        ASSERT_EQ(copied[i], original[i]);
    }
    // Ensure it's a deep copy
    if (!original.empty())
    {
        copied[0] = static_cast<TypeParam>(100);
        ASSERT_NE(copied[0], original[0]);
    }
}

TYPED_TEST(VectorTest, MoveConstructor)
{
    Vector<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    auto original_data = original.data();
    auto original_size = original.size();
    auto original_capacity = original.capacity();

    Vector<TypeParam> moved = std::move(original);

    ASSERT_EQ(moved.size(), original_size);
    ASSERT_EQ(moved.capacity(), original_capacity);
    ASSERT_EQ(moved.data(), original_data);

    // Original vector should be in a valid but empty state
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.data(), nullptr);
}

TYPED_TEST(VectorTest, CopyAssignment)
{
    Vector<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vector<TypeParam> assigned;
    assigned = original;

    ASSERT_EQ(assigned.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i)
    {
        ASSERT_EQ(assigned[i], original[i]);
    }
    // Ensure it's a deep copy
    if (!original.empty())
    {
        assigned[0] = static_cast<TypeParam>(99);
        ASSERT_NE(assigned[0], original[0]);
    }

    // Self-assignment
    assigned = assigned;
    ASSERT_EQ(assigned.size(), 3);
}

TYPED_TEST(VectorTest, MoveAssignment)
{
    Vector<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    auto original_data = original.data();
    auto original_size = original.size();
    Vector<TypeParam> moved;
    moved = std::move(original);

    ASSERT_EQ(moved.size(), original_size);
    ASSERT_EQ(moved.data(), original_data);
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.data(), nullptr);

    // Self-assignment
    moved = std::move(moved);
    ASSERT_EQ(moved.size(), original_size);
}

TYPED_TEST(VectorTest, At)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(10), static_cast<TypeParam>(20)};
    ASSERT_EQ(vec.at(0), static_cast<TypeParam>(10));
    ASSERT_EQ(vec.at(1), static_cast<TypeParam>(20));
    ASSERT_THROW((void)vec.at(2), std::out_of_range); // Check for out-of-bounds
}

TYPED_TEST(VectorTest, OperatorSquareBrackets)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(10)};
    ASSERT_EQ(vec[0], static_cast<TypeParam>(10));
    vec[0] = static_cast<TypeParam>(100);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(100));
}

TYPED_TEST(VectorTest, FrontBack)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    ASSERT_EQ(vec.front(), static_cast<TypeParam>(1));
    ASSERT_EQ(vec.back(), static_cast<TypeParam>(3));
}

TYPED_TEST(VectorTest, Data)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1)};
    ASSERT_NE(vec.data(), nullptr);
    ASSERT_EQ(*vec.data(), static_cast<TypeParam>(1));
}

TYPED_TEST(VectorTest, BeginEnd)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2)};
    auto it = vec.begin();
    ASSERT_EQ(*it, static_cast<TypeParam>(1));
    ++it;
    ASSERT_EQ(*it, static_cast<TypeParam>(2));
    ++it;
    ASSERT_EQ(it, vec.end());
}

TYPED_TEST(VectorTest, Empty)
{
    Vector<TypeParam> vec;
    ASSERT_TRUE(vec.empty());
    vec.pushBack(static_cast<TypeParam>(1));
    ASSERT_FALSE(vec.empty());
}

TYPED_TEST(VectorTest, Size)
{
    Vector<TypeParam> vec;
    ASSERT_EQ(vec.size(), 0);
    vec.pushBack(static_cast<TypeParam>(1));
    ASSERT_EQ(vec.size(), 1);
}

TYPED_TEST(VectorTest, Capacity)
{
    Vector<TypeParam> vec;
    ASSERT_EQ(vec.capacity(), 0);
    vec.reserve(10);
    ASSERT_GE(vec.capacity(), 10);
}

TYPED_TEST(VectorTest, Clear)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2)};
    vec.clear();
    ASSERT_TRUE(vec.empty());
    ASSERT_EQ(vec.size(), 0);
    // Capacity remains unchanged
}

TYPED_TEST(VectorTest, insert)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(3)};
    vec.insert(vec.begin() + 1, static_cast<TypeParam>(2));
    ASSERT_EQ(vec.size(), 3);
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
}

TYPED_TEST(VectorTest, erase)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    vec.erase(vec.begin() + 1);
    ASSERT_EQ(vec.size(), 2);
    ASSERT_EQ(vec[1], static_cast<TypeParam>(3));
}

TYPED_TEST(VectorTest, pushBack)
{
    Vector<TypeParam> vec;
    vec.pushBack(static_cast<TypeParam>(1));
    ASSERT_EQ(vec.size(), 1);
    ASSERT_EQ(vec.back(), static_cast<TypeParam>(1));
}

TYPED_TEST(VectorTest, popBack)
{
    Vector<TypeParam> vec = {static_cast<TypeParam>(1)};
    vec.popBack();
    ASSERT_TRUE(vec.empty());
}

TYPED_TEST(VectorTest, resize)
{
     Vector<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2)};
    vec.resize(5);
    ASSERT_EQ(vec.size(), 5);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(1));
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
    for (size_t i = 2; i < 5; ++i)
    {
        ASSERT_EQ(vec[i], static_cast<TypeParam>(0));
    }
    vec.resize(2);
    ASSERT_EQ(vec.size(), 2);
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
}

TYPED_TEST(VectorTest, swap)
{
    Vector<TypeParam> vec1 = {static_cast<TypeParam>(1), static_cast<TypeParam>(2)};
    Vector<TypeParam> vec2 = {static_cast<TypeParam>(3), static_cast<TypeParam>(4), static_cast<TypeParam>(5)};
    vec1.swap(vec2);
    ASSERT_EQ(vec1.size(), 3);
    ASSERT_EQ(vec2.size(), 2);
}

TYPED_TEST(VectorTest, ShrinkToFit)
{
    Vector<TypeParam> vec;
    vec.reserve(20);
    vec.pushBack(static_cast<TypeParam>(1));
    vec.pushBack(static_cast<TypeParam>(2));
    ASSERT_GE(vec.capacity(), 20);
    vec.shrinkToFit();
    ASSERT_EQ(vec.capacity(), 2);
    ASSERT_EQ(vec.size(), 2);
}

TYPED_TEST(VectorTest, SubVector)
{
    Vector<TypeParam> vec = {0, 1, 2, 3, 4, 5};
    Vector<TypeParam> sub = vec.subVector(2, 3);
    ASSERT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], static_cast<TypeParam>(2));
    EXPECT_EQ(sub[1], static_cast<TypeParam>(3));
    EXPECT_EQ(sub[2], static_cast<TypeParam>(4));

    // Test out of bounds
    ASSERT_THROW(vec.subVector(4, 3), std::out_of_range);
}

TYPED_TEST(VectorTest, EraseRange)
{
    Vector<TypeParam> vec = {0, 1, 2, 3, 4, 5};
    auto it = vec.erase(vec.begin() + 2, vec.begin() + 4); // Erase 2 and 3
    ASSERT_EQ(vec.size(), 4);
    EXPECT_EQ(vec[1], static_cast<TypeParam>(1));
    EXPECT_EQ(vec[2], static_cast<TypeParam>(4));
    EXPECT_EQ(*it, static_cast<TypeParam>(4));
}

TYPED_TEST(VectorTest, EmptyVectorAccess)
{
    Vector<TypeParam> vec;
    ASSERT_THROW((void)vec.front(), std::out_of_range);
    ASSERT_THROW((void)vec.back(), std::out_of_range);
    ASSERT_NO_THROW(vec.popBack()); // Should be safe
}
