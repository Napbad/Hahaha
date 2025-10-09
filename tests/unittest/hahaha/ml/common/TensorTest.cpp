// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

#include <gtest/gtest.h>

#include <ds/Vector.h>
#include <ml/Tensor.h>

using namespace hahaha;
using namespace hahaha::ml;
using namespace hahaha::common;

// Test fixture for Tensor
template <typename T> class TensorTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Common setup for tests
    }

    void TearDown() override
    {
        // Common teardown for tests
    }

    // Helper function to check if two floats are approximately equal
    static bool isApproxEqual(const T a, const T b, const T tolerance = static_cast<T>(0.0001))
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            return std::abs(a - b) < tolerance;
        }
        else
        {
            return a == b;
        }
    }
};

// Define type-parameterized tests
using TestTypes = ::testing::Types<int, f32, f64>;
TYPED_TEST_SUITE(TensorTest, TestTypes);

// Test default constructor
TYPED_TEST(TensorTest, DefaultConstructor)
{
    Tensor<TypeParam> tensor;
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.size(), 0);
}

// Test constructor with shape only
TYPED_TEST(TensorTest, ShapeConstructor)
{
    Tensor<TypeParam> tensor({2, 3});
    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_FALSE(tensor.empty());

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test constructor with shape and data
TYPED_TEST(TensorTest, ShapeDataConstructor)
{
    Tensor<TypeParam> tensor(
        {2, 2},
        {static_cast<TypeParam>(1), static_cast<TypeParam>(2), static_cast<TypeParam>(3), static_cast<TypeParam>(4)});

    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(), 4);
}

// Test 1D tensor
TYPED_TEST(TensorTest, OneDimensionalTensor)
{
    Tensor<TypeParam> tensor({5});
    EXPECT_EQ(tensor.dim(), 1);
    EXPECT_EQ(tensor.size(), 5);

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 5);
}

// Test 3D tensor
TYPED_TEST(TensorTest, ThreeDimensionalTensor)
{
    Tensor<TypeParam> tensor({2, 3, 4});
    EXPECT_EQ(tensor.dim(), 3);
    EXPECT_EQ(tensor.size(), 24);

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test fill method
TYPED_TEST(TensorTest, FillMethod)
{
    Tensor<TypeParam> tensor({3, 3});
    tensor.fill(static_cast<TypeParam>(5));

    for (const auto& val : tensor)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test index calculation
TYPED_TEST(TensorTest, IndexCalculation)
{
    Tensor<TypeParam> tensor({2, 3});

    // Test valid indices
    auto idx1 = tensor.index({0, 0});
    EXPECT_TRUE(idx1.isOk());
    EXPECT_EQ(idx1.unwrap(), 0);

    auto idx2 = tensor.index({0, 1});
    EXPECT_TRUE(idx2.isOk());
    EXPECT_EQ(idx2.unwrap(), 1);

    auto idx3 = tensor.index({1, 0});
    EXPECT_TRUE(idx3.isOk());
    EXPECT_EQ(idx3.unwrap(), 3);

    auto idx4 = tensor.index({1, 2});
    EXPECT_TRUE(idx4.isOk());
    EXPECT_EQ(idx4.unwrap(), 5);
}

// Test index out of bounds
TYPED_TEST(TensorTest, IndexOutOfBounds)
{
    Tensor<TypeParam> tensor({2, 3});

    auto idx1 = tensor.index({2, 0});
    EXPECT_TRUE(idx1.isErr());

    auto idx2 = tensor.index({0, 3});
    EXPECT_TRUE(idx2.isErr());

    auto idx3 = tensor.index({0}); // Wrong number of indices
    EXPECT_TRUE(idx3.isErr());
}

// Test set and at methods
TYPED_TEST(TensorTest, SetAndAtMethods)
{
    Tensor<TypeParam> tensor({2, 3});

    // Set values
    auto res1 = tensor.set({0, 0}, static_cast<TypeParam>(1));
    EXPECT_TRUE(res1.isOk());

    auto res2 = tensor.set({0, 1}, static_cast<TypeParam>(2));
    EXPECT_TRUE(res2.isOk());

    auto res3 = tensor.set({1, 2}, static_cast<TypeParam>(6));
    EXPECT_TRUE(res3.isOk());

    // Get values using at
    auto val1 = tensor.at({0, 0});
    EXPECT_TRUE(val1.isOk());
    EXPECT_EQ(val1.unwrap().first(), static_cast<TypeParam>(1));

    auto val2 = tensor.at({0, 1});
    EXPECT_TRUE(val2.isOk());
    EXPECT_EQ(val2.unwrap().first(), static_cast<TypeParam>(2));

    auto val3 = tensor.at({1, 2});
    EXPECT_TRUE(val3.isOk());
    EXPECT_EQ(val3.unwrap().first(), static_cast<TypeParam>(6));
}

// Test element-wise addition
TYPED_TEST(TensorTest, ElementWiseAddition)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(2));

    auto result = tensor1 + tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test element-wise subtraction
TYPED_TEST(TensorTest, ElementWiseSubtraction)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(5));
    tensor2.fill(static_cast<TypeParam>(2));

    auto result = tensor1 - tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test element-wise multiplication
TYPED_TEST(TensorTest, ElementWiseMultiplication)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(4));

    auto result = tensor1 * tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

// Test element-wise division
TYPED_TEST(TensorTest, ElementWiseDivision)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(12));
    tensor2.fill(static_cast<TypeParam>(4));

    auto result = tensor1 / tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test scalar addition
TYPED_TEST(TensorTest, ScalarAddition)
{
    Tensor<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(3));

    auto result = tensor + static_cast<TypeParam>(2);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test scalar multiplication
TYPED_TEST(TensorTest, ScalarMultiplication)
{
    Tensor<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(3));

    auto result = tensor * static_cast<TypeParam>(4);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

// Test scalar subtraction
TYPED_TEST(TensorTest, ScalarSubtraction)
{
    Tensor<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(10));

    auto result = tensor - static_cast<TypeParam>(3);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(7));
    }
}

// Test scalar division
TYPED_TEST(TensorTest, ScalarDivision)
{
    Tensor<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(12));

    auto result = tensor / static_cast<TypeParam>(4);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test dot product
TYPED_TEST(TensorTest, DotProduct)
{
    Tensor<TypeParam> tensor1({3});
    Tensor<TypeParam> tensor2({3});

    tensor1.set({0}, static_cast<TypeParam>(1));
    tensor1.set({1}, static_cast<TypeParam>(2));
    tensor1.set({2}, static_cast<TypeParam>(3));

    tensor2.set({0}, static_cast<TypeParam>(4));
    tensor2.set({1}, static_cast<TypeParam>(5));
    tensor2.set({2}, static_cast<TypeParam>(6));

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    auto result = tensor1.dot(tensor2);
    EXPECT_EQ(result, static_cast<TypeParam>(32));
}

// Test compound assignment operators
TYPED_TEST(TensorTest, CompoundAddition)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(2));

    tensor1 += tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

TYPED_TEST(TensorTest, CompoundSubtraction)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(5));
    tensor2.fill(static_cast<TypeParam>(2));

    tensor1 -= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

TYPED_TEST(TensorTest, CompoundMultiplication)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(4));

    tensor1 *= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

TYPED_TEST(TensorTest, CompoundDivision)
{
    Tensor<TypeParam> tensor1({2, 2});
    Tensor<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(12));
    tensor2.fill(static_cast<TypeParam>(4));

    tensor1 /= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test fromVector static method
TYPED_TEST(TensorTest, FromVector)
{
    ds::Vector<TypeParam> vec;
    vec.pushBack(static_cast<TypeParam>(1));
    vec.pushBack(static_cast<TypeParam>(2));
    vec.pushBack(static_cast<TypeParam>(3));

    auto tensor = Tensor<TypeParam>::fromVector(vec);

    EXPECT_EQ(tensor.size(), 3);
    EXPECT_EQ(tensor.dim(), 1);

    auto val0 = tensor.at({0});
    EXPECT_TRUE(val0.isOk());
    EXPECT_EQ(val0.unwrap().first(), static_cast<TypeParam>(1));

    auto val2 = tensor.at({2});
    EXPECT_TRUE(val2.isOk());
    EXPECT_EQ(val2.unwrap().first(), static_cast<TypeParam>(3));
}

// Test copy from tensor
TYPED_TEST(TensorTest, CopyFromTensor)
{
    Tensor<TypeParam> source({2, 2});
    source.fill(static_cast<TypeParam>(5));

    Tensor<TypeParam> dest({2, 2});
    auto result = dest.copy(source);

    EXPECT_TRUE(result.isOk());

    for (const auto& val : dest)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test copy from vector
TYPED_TEST(TensorTest, CopyFromVector)
{
    ds::Vector<TypeParam> vec;
    vec.pushBack(static_cast<TypeParam>(1));
    vec.pushBack(static_cast<TypeParam>(2));
    vec.pushBack(static_cast<TypeParam>(3));
    vec.pushBack(static_cast<TypeParam>(4));

    Tensor<TypeParam> tensor({2, 2});
    auto result = tensor.copy(vec);

    EXPECT_TRUE(result.isOk());

    auto val0 = tensor.at({0, 0});
    EXPECT_EQ(val0.unwrap().first(), static_cast<TypeParam>(1));

    auto val3 = tensor.at({1, 1});
    EXPECT_EQ(val3.unwrap().first(), static_cast<TypeParam>(4));
}

// Test copy with mismatched shapes
TYPED_TEST(TensorTest, CopyMismatchedShapes)
{
    Tensor<TypeParam> source({2, 3});
    Tensor<TypeParam> dest({3, 2});

    auto result = dest.copy(source);
    EXPECT_TRUE(result.isErr());
}

// Test iterators
TYPED_TEST(TensorTest, Iterators)
{
    Tensor<TypeParam> tensor({3});
    tensor.set({0}, static_cast<TypeParam>(1));
    tensor.set({1}, static_cast<TypeParam>(2));
    tensor.set({2}, static_cast<TypeParam>(3));

    sizeT count = 0;
    TypeParam sum = 0;

    for (auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        sum += *it;
        count++;
    }

    EXPECT_EQ(count, 3);
    EXPECT_EQ(sum, static_cast<TypeParam>(6));
}

// Test first method
TYPED_TEST(TensorTest, FirstMethod)
{
    Tensor<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(0));
    tensor.set({0, 0}, static_cast<TypeParam>(42));

    EXPECT_EQ(tensor.first(), static_cast<TypeParam>(42));
}

// Test shape mismatch in operations
TYPED_TEST(TensorTest, ShapeMismatchInAddition)
{
    Tensor<TypeParam> tensor1({2, 4});
    Tensor<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 + tensor2, std::runtime_error);
}

TYPED_TEST(TensorTest, ShapeMismatchInSubtraction)
{
    Tensor<TypeParam> tensor1({2, 4});
    Tensor<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 - tensor2, std::runtime_error);
}

TYPED_TEST(TensorTest, ShapeMismatchInMultiplication)
{
    Tensor<TypeParam> tensor1({2, 4});
    Tensor<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 * tensor2, std::runtime_error);
}

TYPED_TEST(TensorTest, ShapeMismatchInDivision)
{
    Tensor<TypeParam> tensor1({2, 4});
    Tensor<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 / tensor2, std::runtime_error);
}

// Floating point specific tests
class TensorFloatTest : public ::testing::Test
{
};

TEST_F(TensorFloatTest, DivisionByZeroElement)
{
    Tensor<f32> tensor1({2, 2});
    Tensor<f32> tensor2({2, 2});

    tensor1.fill(10.0f);
    tensor2.fill(0.0f);

    EXPECT_THROW(tensor1 / tensor2, std::runtime_error);
}

TEST_F(TensorFloatTest, DivisionByZeroScalar)
{
    Tensor<f32> tensor({2, 2});
    tensor.fill(10.0f);

    EXPECT_THROW(tensor / 0.0f, std::runtime_error);
}

// Test high-dimensional tensors
TEST_F(TensorFloatTest, FourDimensionalTensor)
{
    Tensor<f32> tensor({2, 3, 4, 5});
    EXPECT_EQ(tensor.dim(), 4);
    EXPECT_EQ(tensor.size(), 120);
}

// Test edge cases
TEST_F(TensorFloatTest, ScalarTensor)
{
    Tensor<f32> tensor({0}, {42.0f});
    EXPECT_EQ(tensor.dim(), 1);
}

TEST_F(TensorFloatTest, LargeValues)
{
    Tensor<f32> tensor({2, 2});
    tensor.fill(1e30f);

    auto result = tensor + tensor;
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 2e30f);
    }
}

// Test rawData method
TEST_F(TensorFloatTest, RawDataMethod)
{
    Tensor<f32> tensor({3});
    tensor.set({0}, 1.0f);
    tensor.set({1}, 2.0f);
    tensor.set({2}, 3.0f);

    auto data = tensor.rawData();
    EXPECT_EQ(data.size(), 3);
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
}

// Test complex indexing scenarios
TEST_F(TensorFloatTest, ComplexIndexing3D)
{
    Tensor<f32> tensor({2, 3, 4});

    // Test corner cases
    auto idx1 = tensor.index({0, 0, 0});
    EXPECT_TRUE(idx1.isOk());
    EXPECT_EQ(idx1.unwrap(), 0);

    auto idx2 = tensor.index({1, 2, 3});
    EXPECT_TRUE(idx2.isOk());
    EXPECT_EQ(idx2.unwrap(), 23); // (1*3*4) + (2*4) + 3 = 12 + 8 + 3 = 23
}

// Test sequential operations
TEST_F(TensorFloatTest, SequentialOperations)
{
    Tensor<f32> tensor({3, 3});
    tensor.fill(2.0f);

    auto result = ((tensor + 1.0f) * 2.0f) - 3.0f;

    // (2 + 1) * 2 - 3 = 3 * 2 - 3 = 6 - 3 = 3
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

// Test empty tensor
TEST_F(TensorFloatTest, EmptyTensorCheck)
{
    Tensor<f32> empty_tensor;
    EXPECT_TRUE(empty_tensor.empty());

    Tensor<f32> non_empty({2, 2});
    EXPECT_FALSE(non_empty.empty());
}

// Test operator() access method
TEST_F(TensorFloatTest, OperatorParenthesesAccess)
{
    Tensor<f32> tensor({3, 3});
    ds::Vector<sizeT> idx1{0, 0};
    ds::Vector<sizeT> idx2{1, 2};
    ds::Vector<sizeT> idx3{2, 2};

    tensor(idx1) = 1.0f;
    tensor(idx2) = 5.0f;
    tensor(idx3) = 9.0f;

    EXPECT_FLOAT_EQ(tensor(idx1), 1.0f);
    EXPECT_FLOAT_EQ(tensor(idx2), 5.0f);
    EXPECT_FLOAT_EQ(tensor(idx3), 9.0f);
}

// Test variadic operator() access
TYPED_TEST(TensorTest, VariadicOperatorParenthesesAccess)
{
    // 1D Vector
    Tensor<TypeParam> vec({3}, {1, 2, 3});
    EXPECT_EQ(vec(0ul), static_cast<TypeParam>(1));
    EXPECT_EQ(vec(2ul), static_cast<TypeParam>(3));
    vec(1ul) = static_cast<TypeParam>(22);
    EXPECT_EQ(vec(1ul), static_cast<TypeParam>(22));

    // 2D Matrix
    Tensor<TypeParam> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(mat(0ul, 1ul), static_cast<TypeParam>(2));
    EXPECT_EQ(mat(1ul, 2ul), static_cast<TypeParam>(6));
    mat(1ul, 0ul) = static_cast<TypeParam>(44);
    EXPECT_EQ(mat(1ul, 0ul), static_cast<TypeParam>(44));

    // 3D Tensor
    Tensor<TypeParam> t3d({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_EQ(t3d(0ul, 1ul, 0ul), static_cast<TypeParam>(3));
    EXPECT_EQ(t3d(1ul, 1ul, 1ul), static_cast<TypeParam>(8));
    t3d(1ul, 0ul, 1ul) = static_cast<TypeParam>(66);
    EXPECT_EQ(t3d(1ul, 0ul, 1ul), static_cast<TypeParam>(66));

    // Incorrect number of indices
    EXPECT_THROW(mat(0ul), std::runtime_error);
    EXPECT_THROW(mat(0ul, 1ul, 2ul), std::runtime_error);
}

// Test reshape method
TYPED_TEST(TensorTest, Reshape)
{
    Tensor<TypeParam> tensor({2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    
    // Valid reshape
    tensor.reshape({3, 4});
    const auto& new_shape = tensor.shape();
    EXPECT_EQ(new_shape.size(), 2);
    EXPECT_EQ(new_shape[0], 3);
    EXPECT_EQ(new_shape[1], 4);

    // Check that data is unchanged
    EXPECT_EQ(tensor[0], static_cast<TypeParam>(1));
    EXPECT_EQ(tensor[11], static_cast<TypeParam>(12));

    // Another valid reshape
    tensor.reshape({2, 2, 3});
    const auto& new_shape_3d = tensor.shape();
    EXPECT_EQ(new_shape_3d.size(), 3);
    EXPECT_EQ(new_shape_3d[0], 2);
    EXPECT_EQ(new_shape_3d[1], 2);
    EXPECT_EQ(new_shape_3d[2], 3);

    // Invalid reshape (incorrect size)
    EXPECT_THROW(tensor.reshape({3, 5}), std::runtime_error);
}

// Test matmul method
TYPED_TEST(TensorTest, Matmul)
{
    // Standard matrix multiplication
    Tensor<TypeParam> mat1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<TypeParam> mat2({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<TypeParam> result = mat1.matmul(mat2);

    const auto& result_shape = result.shape();
    EXPECT_EQ(result_shape.size(), 2);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 2);

    EXPECT_EQ(result(0, 0), 58);   // 1*7 + 2*9 + 3*11
    EXPECT_EQ(result(0, 1), 64);   // 1*8 + 2*10 + 3*12
    EXPECT_EQ(result(1, 0), 139);  // 4*7 + 5*9 + 6*11
    EXPECT_EQ(result(1, 1), 154);  // 4*8 + 5*10 + 6*12

    // Matrix-vector multiplication
    Tensor<TypeParam> mat_vec({3, 1}, {1, 2, 3});
    Tensor<TypeParam> result_vec = mat1.matmul(mat_vec);
    EXPECT_EQ(result_vec(0, 0), 14); // 1*1 + 2*2 + 3*3
    EXPECT_EQ(result_vec(1, 0), 32); // 4*1 + 5*2 + 6*3

    // Incompatible shapes
    Tensor<TypeParam> mat3({2, 2});
    EXPECT_THROW(mat1.matmul(mat3), std::runtime_error);
}

// Test factory methods
TYPED_TEST(TensorTest, FactoryMethods)
{
    // Zeros
    Tensor<TypeParam> zeros = Tensor<TypeParam>::zeros({2, 3});
    EXPECT_EQ(zeros.shape()[0], 2);
    EXPECT_EQ(zeros.shape()[1], 3);
    for (const auto& val : zeros)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(0));
    }

    // Ones
    Tensor<TypeParam> ones = Tensor<TypeParam>::ones({3, 2});
    EXPECT_EQ(ones.shape()[0], 3);
    EXPECT_EQ(ones.shape()[1], 2);
    for (const auto& val : ones)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(1));
    }

    // Rand
    Tensor<TypeParam> random = Tensor<TypeParam>::rand({4, 4});
    EXPECT_EQ(random.shape()[0], 4);
    EXPECT_EQ(random.shape()[1], 4);
    for (const auto& val : random)
    {
        EXPECT_GE(val, static_cast<TypeParam>(0));
        EXPECT_LE(val, static_cast<TypeParam>(1));
    }
}

// Test scalar tensor operations
TYPED_TEST(TensorTest, ScalarTensorOperations)
{
    Tensor<TypeParam> scalar1({}, {5});
    Tensor<TypeParam> scalar2({}, {3});

    // Addition
    Tensor<TypeParam> sum = scalar1 + scalar2;
    EXPECT_EQ(sum.dim(), 0);
    EXPECT_EQ(sum.first(), static_cast<TypeParam>(8));

    // Multiplication
    Tensor<TypeParam> prod = scalar1 * scalar2;
    EXPECT_EQ(prod.dim(), 0);
    EXPECT_EQ(prod.first(), static_cast<TypeParam>(15));

    std::cout << "scalar1: " << &scalar1 << std::endl;
    std::cout << "scalar2: " << &scalar2 << std::endl;
    std::cout << "sum: " << &sum << std::endl;
    std::cout << "prod: " << &prod << std::endl;
}

TYPED_TEST(TensorTest, ReshapeDataIntegrity)
{
    Tensor<TypeParam> tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    tensor.reshape({4, 2});

    EXPECT_EQ(tensor(0, 0), static_cast<TypeParam>(1));
    EXPECT_EQ(tensor(0, 1), static_cast<TypeParam>(2));
    EXPECT_EQ(tensor(1, 0), static_cast<TypeParam>(3));
    EXPECT_EQ(tensor(1, 1), static_cast<TypeParam>(4));
    EXPECT_EQ(tensor(3, 1), static_cast<TypeParam>(8));
}

TYPED_TEST(TensorTest, MatmulEdgeCases)
{
    // Identity matrix
    Tensor<TypeParam> mat({2, 2}, {1, 0, 0, 1});
    Tensor<TypeParam> other({2, 2}, {5, 6, 7, 8});
    Tensor<TypeParam> result = mat.matmul(other);
    EXPECT_TRUE(result == other);

    // Non-commutative
    Tensor<TypeParam> mat1({2, 2}, {1, 2, 3, 4});
    Tensor<TypeParam> mat2({2, 2}, {5, 6, 7, 8});
    Tensor<TypeParam> result1 = mat1.matmul(mat2);
    Tensor<TypeParam> result2 = mat2.matmul(mat1);
    EXPECT_FALSE(result1 == result2);
}


//
// // Test tensor slicing with at() method
// TEST_F(TensorFloatTest, TensorSlicing3D)
// {
//     Tensor<f32> tensor({2, 3, 4});
//     tensor.fill(5.0f);
//
//     // Get a 2D slice by fixing first dimension
//     auto slice = tensor.at({0});
//     EXPECT_TRUE(slice.isOk());
//
//     auto result = slice.unwrap();
//     EXPECT_EQ(result.dim(), 2);
//     EXPECT_EQ(result.size(), 12); // 3 * 4
//
//     for (const auto& val : result)
//     {
//         EXPECT_FLOAT_EQ(val, 5.0f);
//     }
// }
//
// // Test tensor slicing with at() method - 2D to 1D
// TEST_F(TensorFloatTest, TensorSlicing2DTo1D)
// {
//     Tensor<f32> tensor({3, 4});
//
//     // Fill with sequential values
//     for (sizeT i = 0; i < 3; ++i)
//     {
//         for (sizeT j = 0; j < 4; ++j)
//         {
//             tensor.set({i, j}, static_cast<f32>(i * 4 + j));
//         }
//     }
//
//     // Get the second row
//     auto row = tensor.at({1});
//     EXPECT_TRUE(row.isOk());
//
//     auto result = row.unwrap();
//     EXPECT_EQ(result.dim(), 1);
//     EXPECT_EQ(result.size(), 4);
//
//     EXPECT_FLOAT_EQ(result.at({0}).unwrap().first(), 4.0f);
//     EXPECT_FLOAT_EQ(result.at({1}).unwrap().first(), 5.0f);
//     EXPECT_FLOAT_EQ(result.at({2}).unwrap().first(), 6.0f);
//     EXPECT_FLOAT_EQ(result.at({3}).unwrap().first(), 7.0f);
// }

// Test at() with wrong number of indices
TEST_F(TensorFloatTest, AtMethodTooManyIndices)
{
    Tensor<f32> tensor({2, 3});

    auto result = tensor.at({0, 1, 2});
    EXPECT_TRUE(result.isErr());
}

// Test at() with empty indices on non-empty tensor
TEST_F(TensorFloatTest, AtMethodEmptyIndices)
{
    Tensor<f32> tensor({2, 3});

    auto result = tensor.at({});
    EXPECT_TRUE(result.isErr());
}

// Test set() with wrong number of indices
TEST_F(TensorFloatTest, SetMethodTooManyIndices)
{
    Tensor<f32> tensor({2, 3});

    auto result = tensor.set({0, 1, 2}, 5.0f);
    EXPECT_TRUE(result.isErr());
}

// Test set() with empty indices on non-empty tensor
TEST_F(TensorFloatTest, SetMethodEmptyIndices)
{
    Tensor<f32> tensor({2, 3});

    auto result = tensor.set({}, 5.0f);
    EXPECT_TRUE(result.isErr());
}

// Test copy with Vector size mismatch
TEST_F(TensorFloatTest, CopyFromVecSizeMismatch)
{
    ds::Vector<f32> vec;
    vec.pushBack(1.0f);
    vec.pushBack(2.0f);
    vec.pushBack(3.0f);

    Tensor<f32> tensor({2, 2}); // Size 4, but vec has size 3
    auto result = tensor.copy(vec);

    EXPECT_TRUE(result.isErr());
}

// Test computeSize with 0D tensor
TEST_F(TensorFloatTest, ZeroDimensionalTensor)
{
    Tensor<f32> scalar({1});
    EXPECT_EQ(scalar.dim(), 1);
    EXPECT_EQ(scalar.size(), 1);

    scalar.fill(42.0f);
    EXPECT_FLOAT_EQ(scalar.first(), 42.0f);
}

// Test large tensor operations
TEST_F(TensorFloatTest, LargeTensorOperations)
{
    Tensor<f32> tensor1({10, 10, 10});
    Tensor<f32> tensor2({10, 10, 10});

    tensor1.fill(2.0f);
    tensor2.fill(3.0f);

    auto sum = tensor1 + tensor2;
    auto diff = tensor1 - tensor2;
    auto prod = tensor1 * tensor2;
    auto quot = tensor2 / tensor1;

    EXPECT_EQ(sum.size(), 1000);
    EXPECT_EQ(diff.size(), 1000);
    EXPECT_EQ(prod.size(), 1000);
    EXPECT_EQ(quot.size(), 1000);

    for (const auto& val : sum)
    {
        EXPECT_FLOAT_EQ(val, 5.0f);
    }
    for (const auto& val : diff)
    {
        EXPECT_FLOAT_EQ(val, -1.0f);
    }
    for (const auto& val : prod)
    {
        EXPECT_FLOAT_EQ(val, 6.0f);
    }
    for (const auto& val : quot)
    {
        EXPECT_FLOAT_EQ(val, 1.5f);
    }
}

// Test mixed operations
TEST_F(TensorFloatTest, MixedOperations)
{
    Tensor<f32> a({3, 3});
    Tensor<f32> b({3, 3});

    a.fill(10.0f);
    b.fill(2.0f);

    // (a - b) * 2 + b / 2
    auto result = ((a - b) * 2.0f + b) / 2.0f;

    // (10 - 2) * 2 + 2 / 2 = 8 * 2 + 2 / 2 = 16 + 2 / 2 = 18 / 2 = 9
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 9.0f);
    }
}

// Test compound operations preserve shape
TEST_F(TensorFloatTest, CompoundOperationsPreserveShape)
{
    Tensor<f32> tensor1({2, 3, 4});
    Tensor<f32> tensor2({2, 3, 4});

    tensor1.fill(5.0f);
    tensor2.fill(2.0f);

    tensor1 += tensor2;
    EXPECT_EQ(tensor1.dim(), 3);
    EXPECT_EQ(tensor1.size(), 24);

    const auto& shape = tensor1.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test dot product with zeros
TEST_F(TensorFloatTest, DotProductWithZeros)
{
    Tensor<f32> tensor1({5});
    Tensor<f32> tensor2({5});

    tensor1.fill(5.0f);
    tensor2.fill(0.0f);

    auto result = tensor1.dot(tensor2);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// Test dot product commutative property
TEST_F(TensorFloatTest, DotProductCommutative)
{
    Tensor<f32> tensor1({4});
    Tensor<f32> tensor2({4});

    tensor1.set({0}, 1.0f);
    tensor1.set({1}, 2.0f);
    tensor1.set({2}, 3.0f);
    tensor1.set({3}, 4.0f);

    tensor2.set({0}, 5.0f);
    tensor2.set({1}, 6.0f);
    tensor2.set({2}, 7.0f);
    tensor2.set({3}, 8.0f);

    auto result1 = tensor1.dot(tensor2);
    auto result2 = tensor2.dot(tensor1);

    EXPECT_FLOAT_EQ(result1, result2);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_FLOAT_EQ(result1, 70.0f);
}

// Test index calculation for higher dimensions
TEST_F(TensorFloatTest, IndexCalculation4D)
{
    Tensor<f32> tensor({2, 3, 4, 5});

    // Test corner indices
    auto idx1 = tensor.index({0, 0, 0, 0});
    EXPECT_TRUE(idx1.isOk());
    EXPECT_EQ(idx1.unwrap(), 0);

    auto idx2 = tensor.index({1, 2, 3, 4});
    EXPECT_TRUE(idx2.isOk());
    // (1*3*4*5) + (2*4*5) + (3*5) + 4 = 60 + 40 + 15 + 4 = 119
    EXPECT_EQ(idx2.unwrap(), 119);
}

// Test rawData returns correct data
TEST_F(TensorFloatTest, RawDataReturnsCorrectValues)
{
    Tensor<f32> tensor({2, 3});

    for (sizeT i = 0; i < 6; ++i)
    {
        tensor.set({i / 3, i % 3}, static_cast<f32>(i));
    }

    auto data = tensor.rawData();
    EXPECT_EQ(data.size(), 6);

    for (sizeT i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<f32>(i));
    }
}

// Test fromVector with large vector
TEST_F(TensorFloatTest, FromVectorLarge)
{
    ds::Vector<f32> vec;
    for (sizeT i = 0; i < 100; ++i)
    {
        vec.pushBack(static_cast<f32>(i));
    }

    auto tensor = Tensor<f32>::fromVector(vec);

    EXPECT_EQ(tensor.size(), 100);
    EXPECT_EQ(tensor.dim(), 1);

    for (sizeT i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(tensor.at({i}).unwrap().first(), static_cast<f32>(i));
    }
}

// Test negative values
TEST_F(TensorFloatTest, NegativeValues)
{
    Tensor<f32> tensor({3, 3});
    tensor.fill(-5.0f);

    for (const auto& val : tensor)
    {
        EXPECT_FLOAT_EQ(val, -5.0f);
    }

    auto result = tensor * -2.0f;
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 10.0f);
    }
}

// Test very small values
TEST_F(TensorFloatTest, VerySmallValues)
{
    Tensor<f32> tensor({2, 2});
    tensor.fill(1e-10f);

    auto result = tensor + tensor;
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 2e-10f);
    }
}

// Integer tensor specific tests
class TensorIntTest : public ::testing::Test
{
};

TEST_F(TensorIntTest, IntegerDivision)
{
    Tensor<int> tensor1({2, 2});
    Tensor<int> tensor2({2, 2});

    tensor1.fill(10);
    tensor2.fill(3);

    auto result = tensor1 / tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, 3); // Integer division
    }
}

TEST_F(TensorIntTest, IntegerArithmetic)
{
    Tensor<int> tensor({3});
    tensor.set({0}, 100);
    tensor.set({1}, 200);
    tensor.set({2}, 300);

    auto result = (tensor + 50) * 2 - 100;

    EXPECT_EQ(result.at({0}).unwrap().first(), 200); // (100+50)*2-100 = 200
    EXPECT_EQ(result.at({1}).unwrap().first(), 400); // (200+50)*2-100 = 400
    EXPECT_EQ(result.at({2}).unwrap().first(), 600); // (300+50)*2-100 = 600
}
