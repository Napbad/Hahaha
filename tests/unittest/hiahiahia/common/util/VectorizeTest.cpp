#include <gtest/gtest.h>
#include "common/util/vectorize/vectorize.h"
#include "common/util/vectorize/vectorize_impl.h"
#include <array>
#include <cmath>

using namespace hahaha::vectorize;

class VectorizeTest : public ::testing::Test {
protected:
    static constexpr size_t VEC_SIZE = 8;
    std::array<float, VEC_SIZE> data1{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float, VEC_SIZE> data2{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
};

TEST_F(VectorizeTest, BasicArithmetic) {
    auto vec1 = make_vec(data1);
    auto vec2 = make_vec(data2);

    // Test addition
    auto sum = vec1 + vec2;
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(sum[i], data1[i] + data2[i]);
    }

    // Test subtraction
    auto diff = vec2 - vec1;
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(diff[i], data2[i] - data1[i]);
    }

    // Test multiplication
    auto prod = vec1 * vec2;
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(prod[i], data1[i] * data2[i]);
    }

    // Test division
    auto quot = vec2 / vec1;
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(quot[i], data2[i] / data1[i]);
    }
}

TEST_F(VectorizeTest, ElementWiseOperations) {
    auto vec1 = make_vec(data1);
    
    // Test absolute value
    std::array<float, VEC_SIZE> neg_data{-1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    auto neg_vec = make_vec(neg_data);
    auto abs_vec = neg_vec.abs();
    
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(abs_vec[i], std::abs(neg_data[i]));
    }
}

TEST_F(VectorizeTest, ReductionOperations) {
    auto vec1 = make_vec(data1);
    
    // Test sum
    float expected_sum = 0.0f;
    for (float val : data1) {
        expected_sum += val;
    }
    
    EXPECT_FLOAT_EQ(vec1.sum(), expected_sum);
}

TEST_F(VectorizeTest, LoadStore) {
    std::array<float, VEC_SIZE> buffer;
    auto vec1 = make_vec(data1);
    
    // Test store
    vec1.store(buffer.data());
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], data1[i]);
    }
    
    // Test load
    VecRegister<float, VEC_SIZE> vec2;
    vec2.load(buffer.data());
    for (size_t i = 0; i < VEC_SIZE; ++i) {
        EXPECT_FLOAT_EQ(vec2[i], data1[i]);
    }
}

TEST_F(VectorizeTest, Comparison) {
    auto vec1 = make_vec(data1);
    auto vec2 = make_vec(data1);  // Same data
    auto vec3 = make_vec(data2);  // Different data
    
    EXPECT_TRUE(vec1 == vec2);
    EXPECT_FALSE(vec1 == vec3);
}

TEST_F(VectorizeTest, ScalarConstruction) {
    float scalar = 42.0f;
    auto vec = make_vec(scalar);
    
    for (size_t i = 0; i < SimdTraits<float>::vector_size; ++i) {
        EXPECT_FLOAT_EQ(vec[i], scalar);
    }
} 