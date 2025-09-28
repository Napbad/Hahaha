#include <gtest/gtest.h>
#include "common/util/vectorize/vec.h"
#include <array>
#include <cstddef>

using namespace hahaha::vectorize;

class VecTest : public ::testing::Test {
protected:
    static constexpr std::size_t VEC_SIZE = 4;
    std::array<int, VEC_SIZE> data1{{1, 2, 3, 4}};  // Double braces to avoid narrowing warning
    std::array<int, VEC_SIZE> data2{{5, 6, 7, 8}};
};

TEST_F(VecTest, Construction) {
    // Default construction
    Vec<int, VEC_SIZE> vec1;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec1[i], 0);
    }

    // Scalar construction
    Vec<int, VEC_SIZE> vec2(42);
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec2[i], 42);
    }

    // Array construction
    Vec<int, VEC_SIZE> vec3(data1);
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i]);
    }

    // Variadic construction
    auto vec4 = make_vec(1, 2, 3, 4);
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec4[i], data1[i]);
    }
}

TEST_F(VecTest, VectorArithmetic) {
    Vec<int, VEC_SIZE> vec1(data1);
    Vec<int, VEC_SIZE> vec2(data2);

    // Addition
    auto sum = vec1 + vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(sum[i], data1[i] + data2[i]);
    }

    // Subtraction
    auto diff = vec2 - vec1;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(diff[i], data2[i] - data1[i]);
    }

    // Multiplication
    auto prod = vec1 * vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(prod[i], data1[i] * data2[i]);
    }

    // Division
    auto quot = vec2 / vec1;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(quot[i], data2[i] / data1[i]);
    }
}

TEST_F(VecTest, ScalarArithmetic) {
    Vec<int, VEC_SIZE> vec(data1);
    const int scalar = 2;

    // Addition
    auto sum = vec + scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(sum[i], data1[i] + scalar);
    }

    // Subtraction
    auto diff = vec - scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(diff[i], data1[i] - scalar);
    }

    // Multiplication
    auto prod = vec * scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(prod[i], data1[i] * scalar);
    }

    // Division
    auto quot = vec / scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(quot[i], data1[i] / scalar);
    }

    // Reverse operations
    auto rev_sum = scalar + vec;
    auto rev_prod = scalar * vec;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(rev_sum[i], scalar + data1[i]);
        EXPECT_EQ(rev_prod[i], scalar * data1[i]);
    }
}

TEST_F(VecTest, CompoundAssignment) {
    Vec<int, VEC_SIZE> vec1(data1);
    Vec<int, VEC_SIZE> vec2(data2);
    const int scalar = 2;

    // Vector compound assignment
    auto vec3 = vec1;
    vec3 += vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] + data2[i]);
    }

    vec3 = vec1;
    vec3 -= vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] - data2[i]);
    }

    vec3 = vec1;
    vec3 *= vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] * data2[i]);
    }

    vec3 = vec1;
    vec3 /= vec2;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] / data2[i]);
    }

    // Scalar compound assignment
    vec3 = vec1;
    vec3 += scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] + scalar);
    }

    vec3 = vec1;
    vec3 -= scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] - scalar);
    }

    vec3 = vec1;
    vec3 *= scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] * scalar);
    }

    vec3 = vec1;
    vec3 /= scalar;
    for (std::size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_EQ(vec3[i], data1[i] / scalar);
    }
}

TEST_F(VecTest, Comparison) {
    Vec<int, VEC_SIZE> vec1(data1);
    Vec<int, VEC_SIZE> vec2(data1);  // Same data
    Vec<int, VEC_SIZE> vec3(data2);  // Different data

    EXPECT_TRUE(vec1 == vec2);
    EXPECT_FALSE(vec1 == vec3);
    EXPECT_FALSE(vec1 != vec2);
    EXPECT_TRUE(vec1 != vec3);
}

TEST_F(VecTest, UtilityFunctions) {
    Vec<int, VEC_SIZE> vec(data1);

    // Sum
    EXPECT_EQ(vec.sum(), 10);  // 1 + 2 + 3 + 4

    // Product
    EXPECT_EQ(vec.product(), 24);  // 1 * 2 * 3 * 4

    // Min/Max
    EXPECT_EQ(vec.min(), 1);
    EXPECT_EQ(vec.max(), 4);
}

TEST_F(VecTest, Iterators) {
    Vec<int, VEC_SIZE> vec(data1);
    
    // Test range-based for loop
    std::size_t i = 0;
    for (const auto& val : vec) {
        EXPECT_EQ(val, data1[i++]);
    }

    // Test const iterators
    i = 0;
    for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
        EXPECT_EQ(*it, data1[i++]);
    }
}

TEST_F(VecTest, ExceptionHandling) {
    Vec<int, VEC_SIZE> vec1(data1);
    Vec<int, VEC_SIZE> vec2;  // All zeros

    // Test out of range access
    EXPECT_THROW(vec1[VEC_SIZE], std::out_of_range);

    // Test division by zero
    EXPECT_THROW(vec1 / vec2, std::domain_error);
    EXPECT_THROW(vec1 / 0, std::domain_error);
} 