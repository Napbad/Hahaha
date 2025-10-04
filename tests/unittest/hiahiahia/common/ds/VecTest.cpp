#include "common/ds/Vec.h"

#include "gtest/gtest.h"

using namespace hahaha::common::ds;

// Test fixture for Vec
template <typename T>
class VecTest : public ::testing::Test {
protected:
    // You can add shared setup/teardown logic here if needed
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Common teardown for tests
    }
};

// Define type-parameterized tests
using TestTypes = ::testing::Types<int, hahaha::f32, double>;
TYPED_TEST_SUITE(VecTest, TestTypes);

TYPED_TEST(VecTest, DefaultConstructor) {
    Vec<TypeParam> vec;
    ASSERT_EQ(vec.size(), 0);
    ASSERT_EQ(vec.capacity(), 0);
    ASSERT_TRUE(vec.empty());
}

TYPED_TEST(VecTest, CountConstructor) {
    Vec<TypeParam> vec(5);
    ASSERT_EQ(vec.size(), 5);
    ASSERT_GE(vec.capacity(), 5); // Capacity should be at least 5
    ASSERT_FALSE(vec.empty());
}

TYPED_TEST(VecTest, InitializerListConstructor) {
    Vec<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3),
        static_cast<TypeParam>(4), static_cast<TypeParam>(5)};
    ASSERT_EQ(vec.size(), 5);
    ASSERT_EQ(vec.capacity(), 5);
    ASSERT_FALSE(vec.empty());
    ASSERT_EQ(vec[0], static_cast<TypeParam>(1));
    ASSERT_EQ(vec[4], static_cast<TypeParam>(5));
}

TYPED_TEST(VecTest, IteratorConstructor) {
    Vec<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vec<TypeParam> vec(original.begin(), original.end());
    ASSERT_EQ(vec.size(), 3);
    ASSERT_EQ(vec.capacity(), 3);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(1));
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
    ASSERT_EQ(vec[2], static_cast<TypeParam>(3));
}

TYPED_TEST(VecTest, CopyConstructor) {
    Vec<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vec<TypeParam> copied   = original;
    ASSERT_EQ(copied.size(), 3);
    ASSERT_EQ(copied.capacity(), 3);
    ASSERT_EQ(copied[0], static_cast<TypeParam>(1));
    ASSERT_EQ(copied[1], static_cast<TypeParam>(2));
    ASSERT_EQ(copied[2], static_cast<TypeParam>(3));
    // Ensure deep copy
    copied[0] = static_cast<TypeParam>(10);
    ASSERT_NE(original[0], copied[0]);
}

TYPED_TEST(VecTest, MoveConstructor) {
    Vec<TypeParam> original  = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    TypeParam* original_data = original.begin();
    Vec<TypeParam> moved     = std::move(original);
    ASSERT_EQ(moved.size(), 3);
    ASSERT_EQ(moved.capacity(), 3);
    ASSERT_EQ(moved[0], static_cast<TypeParam>(1));
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.capacity(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(moved.begin(), original_data); // Ensure data pointer is moved
}

TYPED_TEST(VecTest, CopyAssignment) {
    Vec<TypeParam> original = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vec<TypeParam> assigned;
    assigned = original;
    ASSERT_EQ(assigned.size(), 3);
    ASSERT_EQ(assigned.capacity(), 3);
    ASSERT_EQ(assigned[0], static_cast<TypeParam>(1));
}

TYPED_TEST(VecTest, MoveAssignment) {
    Vec<TypeParam> original  = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    TypeParam* original_data = original.begin();
    Vec<TypeParam> assigned;
    assigned = std::move(original);
    ASSERT_EQ(assigned.size(), 3);
    ASSERT_EQ(assigned[0], static_cast<TypeParam>(1));
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.capacity(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(assigned.begin(), original_data);
}

TYPED_TEST(VecTest, PushBackAndEmplaceBack) {
    Vec<TypeParam> vec;
    vec.push_back(static_cast<TypeParam>(1));
    vec.emplace_back(static_cast<TypeParam>(2));
    ASSERT_EQ(vec.size(), 2);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(1));
    ASSERT_EQ(vec[1], static_cast<TypeParam>(2));
    ASSERT_GE(vec.capacity(), 2);
}

TYPED_TEST(VecTest, OperatorAccess) {
    Vec<TypeParam> vec = {static_cast<TypeParam>(10), static_cast<TypeParam>(20), static_cast<TypeParam>(30)};
    ASSERT_EQ(vec[0], static_cast<TypeParam>(10));
    ASSERT_EQ(vec[1], static_cast<TypeParam>(20));
    vec[0] = static_cast<TypeParam>(5);
    ASSERT_EQ(vec[0], static_cast<TypeParam>(5));
}

TYPED_TEST(VecTest, EqualityOperator) {
    Vec<TypeParam> vec1 = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vec<TypeParam> vec2 = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    Vec<TypeParam> vec3 = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(4)};
    Vec<TypeParam> vec4 = {static_cast<TypeParam>(1), static_cast<TypeParam>(2)};

    ASSERT_EQ(vec1, vec2);
    ASSERT_NE(vec1, vec3);
    ASSERT_NE(vec1, vec4);
}

TYPED_TEST(VecTest, Clear) {
    Vec<TypeParam> vec = {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3)};
    vec.clear();
    ASSERT_EQ(vec.size(), 0);
    ASSERT_TRUE(vec.empty());
}

TYPED_TEST(VecTest, Reserve) {
    Vec<TypeParam> vec;
    vec.reserve(10);
    ASSERT_EQ(vec.size(), 0);
    ASSERT_GE(vec.capacity(), 10);

    vec.push_back(static_cast<TypeParam>(1));
    vec.reserve(5); // Should not shrink
    ASSERT_GE(vec.capacity(), 10);
    ASSERT_EQ(vec.size(), 1);
}

TYPED_TEST(VecTest, ShrinkToFit) {
    Vec<TypeParam> vec(10);
    vec.push_back(static_cast<TypeParam>(1));
    vec.push_back(static_cast<TypeParam>(2));
    ASSERT_EQ(vec.size(), 2);
    ASSERT_GE(vec.capacity(), 10);

    vec.shrink_to_fit();
    ASSERT_EQ(vec.size(), 2);
    ASSERT_EQ(vec.capacity(), 2);

    Vec<TypeParam> empty_vec;
    empty_vec.reserve(10);
    empty_vec.shrink_to_fit();
    ASSERT_EQ(empty_vec.size(), 0);
    ASSERT_EQ(empty_vec.capacity(), 0);
}
