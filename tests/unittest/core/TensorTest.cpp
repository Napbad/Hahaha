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

#include <core/TensorData.h>
#include <ds/Vector.h>
#include <gtest/gtest.h>

using namespace hahaha;
using namespace hahaha::ml;
using namespace hahaha::core;

// Test fixture for TensorData
template <typename T> class TensorDataTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // core setup for tests
    }

    void TearDown() override
    {
        // core teardown for tests
    }

    // Helper function to check if two floats are approximately equal
    static bool isApproxEqual(const T a,
                              const T b,
                              const T tolerance = static_cast<T>(0.0001))
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
TYPED_TEST_SUITE(TensorDataTest, TestTypes);

// Test default constructor
TYPED_TEST(TensorDataTest, DefaultConstructor)
{
    TensorData<TypeParam> tensor;
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.size(), 0);
}

// Test constructor with shape only
TYPED_TEST(TensorDataTest, ShapeConstructor)
{
    TensorData<TypeParam> tensor({2, 3});
    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_FALSE(tensor.empty());

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
}

// Test constructor with shape and data
TYPED_TEST(TensorDataTest, ShapeDataConstructor)
{
    TensorData<TypeParam> tensor({2, 2},
                             {static_cast<TypeParam>(1),
                              static_cast<TypeParam>(2),
                              static_cast<TypeParam>(3),
                              static_cast<TypeParam>(4)});

    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(), 4);
}

// Test 1D tensor
TYPED_TEST(TensorDataTest, OneDimensionalTensorData)
{
    TensorData<TypeParam> tensor({5});
    EXPECT_EQ(tensor.dim(), 1);
    EXPECT_EQ(tensor.size(), 5);

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 5);
}

// Test 3D tensor
TYPED_TEST(TensorDataTest, ThreeDimensionalTensorData)
{
    TensorData<TypeParam> tensor({2, 3, 4});
    EXPECT_EQ(tensor.dim(), 3);
    EXPECT_EQ(tensor.size(), 24);

    const auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test fill method
TYPED_TEST(TensorDataTest, FillMethod)
{
    TensorData<TypeParam> tensor({3, 3});
    tensor.fill(static_cast<TypeParam>(5));

    for (const auto& val : tensor)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test index calculation
TYPED_TEST(TensorDataTest, IndexCalculation)
{
    TensorData<TypeParam> tensor({2, 3});

    // Test valid indices
    EXPECT_EQ(tensor.index({0, 0}), 0);
    EXPECT_EQ(tensor.index({0, 1}), 1);
    EXPECT_EQ(tensor.index({1, 0}), 3);
    EXPECT_EQ(tensor.index({1, 2}), 5);
}

// Test index out of bounds
TYPED_TEST(TensorDataTest, IndexOutOfBounds)
{
    TensorData<TypeParam> tensor({2, 3});
    EXPECT_THROW((void) tensor.index({2, 0}), IndexOutOfBoundError);
    EXPECT_THROW((void) tensor.index({0, 3}), IndexOutOfBoundError);
}

// Test set and at methods
TYPED_TEST(TensorDataTest, SetAndAtMethods)
{
    TensorData<TypeParam> tensor({2, 3});

    // Set values
    tensor.set({0, 0}, static_cast<TypeParam>(1));
    tensor.set({0, 1}, static_cast<TypeParam>(2));
    tensor.set({1, 2}, static_cast<TypeParam>(6));

    // Get values using at
    EXPECT_EQ(tensor.at({0, 0}).first(), static_cast<TypeParam>(1));
    EXPECT_EQ(tensor.at({0, 1}).first(), static_cast<TypeParam>(2));
    EXPECT_EQ(tensor.at({1, 2}).first(), static_cast<TypeParam>(6));
}

// Test element-wise addition
TYPED_TEST(TensorDataTest, ElementWiseAddition)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(2));

    auto result = tensor1 + tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test element-wise subtraction
TYPED_TEST(TensorDataTest, ElementWiseSubtraction)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(5));
    tensor2.fill(static_cast<TypeParam>(2));

    auto result = tensor1 - tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test element-wise multiplication
TYPED_TEST(TensorDataTest, ElementWiseMultiplication)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(4));

    auto result = tensor1 * tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

// Test element-wise division
TYPED_TEST(TensorDataTest, ElementWiseDivision)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(12));
    tensor2.fill(static_cast<TypeParam>(4));

    auto result = tensor1 / tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test scalar addition
TYPED_TEST(TensorDataTest, ScalarAddition)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(3));

    auto result = tensor + static_cast<TypeParam>(2);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test scalar multiplication
TYPED_TEST(TensorDataTest, ScalarMultiplication)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(3));

    auto result = tensor * static_cast<TypeParam>(4);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

// Test scalar subtraction
TYPED_TEST(TensorDataTest, ScalarSubtraction)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(10));

    auto result = tensor - static_cast<TypeParam>(3);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(7));
    }
}

// Test scalar division
TYPED_TEST(TensorDataTest, ScalarDivision)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(12));

    auto result = tensor / static_cast<TypeParam>(4);

    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test dot product
TYPED_TEST(TensorDataTest, DotProduct)
{
    TensorData<TypeParam> tensor1({3});
    TensorData<TypeParam> tensor2({3});

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
TYPED_TEST(TensorDataTest, CompoundAddition)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(2));

    tensor1 += tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

TYPED_TEST(TensorDataTest, CompoundSubtraction)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(5));
    tensor2.fill(static_cast<TypeParam>(2));

    tensor1 -= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

TYPED_TEST(TensorDataTest, CompoundMultiplication)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(3));
    tensor2.fill(static_cast<TypeParam>(4));

    tensor1 *= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(12));
    }
}

TYPED_TEST(TensorDataTest, CompoundDivision)
{
    TensorData<TypeParam> tensor1({2, 2});
    TensorData<TypeParam> tensor2({2, 2});

    tensor1.fill(static_cast<TypeParam>(12));
    tensor2.fill(static_cast<TypeParam>(4));

    tensor1 /= tensor2;

    for (const auto& val : tensor1)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(3));
    }
}

// Test fromVector static method
TYPED_TEST(TensorDataTest, FromVector)
{
    ds::Vector<TypeParam> vec;
    vec.pushBack(static_cast<TypeParam>(1));
    vec.pushBack(static_cast<TypeParam>(2));
    vec.pushBack(static_cast<TypeParam>(3));

    auto tensor = TensorData<TypeParam>::fromVector(vec);

    EXPECT_EQ(tensor.size(), 3);
    EXPECT_EQ(tensor.dim(), 1);

    EXPECT_EQ(tensor.at({0}).first(), static_cast<TypeParam>(1));
    EXPECT_EQ(tensor.at({2}).first(), static_cast<TypeParam>(3));
}

// Test copy from tensor
TYPED_TEST(TensorDataTest, CopyFromTensorData)
{
    TensorData<TypeParam> source({2, 2});
    source.fill(static_cast<TypeParam>(5));

    TensorData<TypeParam> dest({2, 2});
    dest.copy(source);

    for (const auto& val : dest)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test copy from vector
TYPED_TEST(TensorDataTest, CopyFromVector)
{
    ds::Vector<TypeParam> vec;
    vec.pushBack(static_cast<TypeParam>(1));
    vec.pushBack(static_cast<TypeParam>(2));
    vec.pushBack(static_cast<TypeParam>(3));
    vec.pushBack(static_cast<TypeParam>(4));

    TensorData<TypeParam> tensor({2, 2});
    tensor.copy(vec);

    EXPECT_EQ(tensor.at({0, 0}).first(), static_cast<TypeParam>(1));
    EXPECT_EQ(tensor.at({1, 1}).first(), static_cast<TypeParam>(4));
}

// Test copy with mismatched shapes
TYPED_TEST(TensorDataTest, CopyMismatchedShapes)
{
    TensorData<TypeParam> source({2, 3});
    TensorData<TypeParam> dest({3, 2});

    EXPECT_THROW(dest.copy(source), TensorDataErr);
}

// Test iterators
TYPED_TEST(TensorDataTest, Iterators)
{
    TensorData<TypeParam> tensor({3});
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
TYPED_TEST(TensorDataTest, FirstMethod)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(0));
    tensor.set({0, 0}, static_cast<TypeParam>(42));

    EXPECT_EQ(tensor.first(), static_cast<TypeParam>(42));
}

// Test shape mismatch in operations
TYPED_TEST(TensorDataTest, ShapeMismatchInAddition)
{
    TensorData<TypeParam> tensor1({2, 4});
    TensorData<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 + tensor2, std::runtime_error);
}

TYPED_TEST(TensorDataTest, ShapeMismatchInSubtraction)
{
    TensorData<TypeParam> tensor1({2, 4});
    TensorData<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 - tensor2, std::runtime_error);
}

TYPED_TEST(TensorDataTest, ShapeMismatchInMultiplication)
{
    TensorData<TypeParam> tensor1({2, 4});
    TensorData<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 * tensor2, std::runtime_error);
}

TYPED_TEST(TensorDataTest, ShapeMismatchInDivision)
{
    TensorData<TypeParam> tensor1({2, 4});
    TensorData<TypeParam> tensor2({3, 2});

    EXPECT_THROW(tensor1 / tensor2, std::runtime_error);
}

// Floating point specific tests
class TensorDataFloatTest : public ::testing::Test
{
};

TEST_F(TensorDataFloatTest, DivisionByZeroElement)
{
    TensorData<f32> tensor1({2, 2});
    TensorData<f32> tensor2({2, 2});

    tensor1.fill(10.0f);
    tensor2.fill(0.0f);

    EXPECT_THROW(tensor1 / tensor2, std::runtime_error);
}

TEST_F(TensorDataFloatTest, DivisionByZeroScalar)
{
    TensorData<f32> tensor({2, 2});
    tensor.fill(10.0f);

    EXPECT_THROW(tensor / 0.0f, std::runtime_error);
}

// Test high-dimensional tensors
TEST_F(TensorDataFloatTest, FourDimensionalTensorData)
{
    TensorData<f32> tensor({2, 3, 4, 5});
    EXPECT_EQ(tensor.dim(), 4);
    EXPECT_EQ(tensor.size(), 120);
}

// Test edge cases
TEST_F(TensorDataFloatTest, ScalarTensorData)
{
    TensorData<f32> tensor({0}, {42.0f});
    EXPECT_EQ(tensor.dim(), 1);
}

TEST_F(TensorDataFloatTest, LargeValues)
{
    TensorData<f32> tensor({2, 2});
    tensor.fill(1e30f);

    auto result = tensor + tensor;
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 2e30f);
    }
}

// Test rawData method
TEST_F(TensorDataFloatTest, RawDataMethod)
{
    TensorData<f32> tensor({3});
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
TEST_F(TensorDataFloatTest, ComplexIndexing3D)
{
    TensorData<f32> tensor({2, 3, 4});

    // Test corner cases
    EXPECT_EQ(tensor.index({0, 0, 0}), 0);
    EXPECT_EQ(tensor.index({1, 2, 3}),
              23); // (1*3*4) + (2*4) + 3 = 12 + 8 + 3 = 23
}

// Test sequential operations
TEST_F(TensorDataFloatTest, SequentialOperations)
{
    TensorData<f32> tensor({3, 3});
    tensor.fill(2.0f);

    auto result = ((tensor + 1.0f) * 2.0f) - 3.0f;

    // (2 + 1) * 2 - 3 = 3 * 2 - 3 = 6 - 3 = 3
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

// Test empty tensor
TEST_F(TensorDataFloatTest, EmptyTensorDataCheck)
{
    TensorData<f32> empty_tensor;
    EXPECT_TRUE(empty_tensor.empty());

    TensorData<f32> non_empty({2, 2});
    EXPECT_FALSE(non_empty.empty());
}

// Test operator() access method
TEST_F(TensorDataFloatTest, OperatorParenthesesAccess)
{
    TensorData<f32> tensor({3, 3});
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
TYPED_TEST(TensorDataTest, VariadicOperatorParenthesesAccess)
{
    // 1D Vector
    TensorData<TypeParam> vec({3}, {1, 2, 3});
    EXPECT_EQ(vec(0ul), static_cast<TypeParam>(1));
    EXPECT_EQ(vec(2ul), static_cast<TypeParam>(3));
    vec(1ul) = static_cast<TypeParam>(22);
    EXPECT_EQ(vec(1ul), static_cast<TypeParam>(22));

    // 2D Matrix
    TensorData<TypeParam> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(mat(0ul, 1ul), static_cast<TypeParam>(2));
    EXPECT_EQ(mat(1ul, 2ul), static_cast<TypeParam>(6));
    mat(1ul, 0ul) = static_cast<TypeParam>(44);
    EXPECT_EQ(mat(1ul, 0ul), static_cast<TypeParam>(44));

    // 3D TensorData
    TensorData<TypeParam> t3d({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_EQ(t3d(0ul, 1ul, 0ul), static_cast<TypeParam>(3));
    EXPECT_EQ(t3d(1ul, 1ul, 1ul), static_cast<TypeParam>(8));
    t3d(1ul, 0ul, 1ul) = static_cast<TypeParam>(66);
    EXPECT_EQ(t3d(1ul, 0ul, 1ul), static_cast<TypeParam>(66));

    // Incorrect number of indices
    EXPECT_THROW(mat(0ul), std::runtime_error);
    EXPECT_THROW(mat(0ul, 1ul, 2ul), std::runtime_error);
}

// Test scalar constructor and conversion
TYPED_TEST(TensorDataTest, ScalarConstructorAndConversion)
{
    TypeParam scalar_value = static_cast<TypeParam>(42);
    TensorData<TypeParam> scalar_tensor(scalar_value);

    EXPECT_EQ(scalar_tensor.dim(), 0);
    EXPECT_EQ(scalar_tensor.size(), 1);
    EXPECT_EQ(scalar_tensor.first(), scalar_value);

    // Test conversion back to scalar
    TypeParam converted_value = static_cast<TypeParam>(scalar_tensor);
    EXPECT_EQ(converted_value, scalar_value);

    // Test print method for scalar
    std::cout << "Testing print for scalar:" << std::endl;
    scalar_tensor.print();
}

// Test scalar assignment
TYPED_TEST(TensorDataTest, ScalarAssignment)
{
    TensorData<TypeParam> scalar_tensor(static_cast<TypeParam>(10));
    EXPECT_EQ(scalar_tensor.first(), static_cast<TypeParam>(10));

    scalar_tensor = static_cast<TypeParam>(25);
    EXPECT_EQ(scalar_tensor.first(), static_cast<TypeParam>(25));
}

// Test transpose method
TYPED_TEST(TensorDataTest, Transpose)
{
    TensorData<TypeParam> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData<TypeParam> transposed = mat.transpose();

    const auto& transposed_shape = transposed.shape();
    EXPECT_EQ(transposed_shape.size(), 2);
    EXPECT_EQ(transposed_shape[0], 3);
    EXPECT_EQ(transposed_shape[1], 2);

    EXPECT_EQ(transposed(0, 0), static_cast<TypeParam>(1));
    EXPECT_EQ(transposed(0, 1), static_cast<TypeParam>(4));
    EXPECT_EQ(transposed(1, 0), static_cast<TypeParam>(2));
    EXPECT_EQ(transposed(1, 1), static_cast<TypeParam>(5));
    EXPECT_EQ(transposed(2, 0), static_cast<TypeParam>(3));
    EXPECT_EQ(transposed(2, 1), static_cast<TypeParam>(6));

    // Test print method for matrix
    std::cout << "Testing print for matrix:" << std::endl;
    mat.print();
    std::cout << "Testing print for transposed matrix:" << std::endl;
    transposed.print();
}

// Test reshape method
TYPED_TEST(TensorDataTest, Reshape)
{
    TensorData<TypeParam> tensor({2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

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
TYPED_TEST(TensorDataTest, Matmul)
{
    // Standard matrix multiplication
    TensorData<TypeParam> mat1({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData<TypeParam> mat2({3, 2}, {7, 8, 9, 10, 11, 12});
    TensorData<TypeParam> result = mat1.matmul(mat2);

    const auto& result_shape = result.shape();
    EXPECT_EQ(result_shape.size(), 2);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 2);

    EXPECT_EQ(result(0, 0), 58);  // 1*7 + 2*9 + 3*11
    EXPECT_EQ(result(0, 1), 64);  // 1*8 + 2*10 + 3*12
    EXPECT_EQ(result(1, 0), 139); // 4*7 + 5*9 + 6*11
    EXPECT_EQ(result(1, 1), 154); // 4*8 + 5*10 + 6*12

    // Matrix-vector multiplication
    TensorData<TypeParam> mat_vec({3, 1}, {1, 2, 3});
    TensorData<TypeParam> result_vec = mat1.matmul(mat_vec);
    EXPECT_EQ(result_vec(0, 0), 14); // 1*1 + 2*2 + 3*3
    EXPECT_EQ(result_vec(1, 0), 32); // 4*1 + 5*2 + 6*3

    // Incompatible shapes
    TensorData<TypeParam> mat3({2, 2});
    EXPECT_THROW(mat1.matmul(mat3), std::runtime_error);
}

// Test factory methods
TYPED_TEST(TensorDataTest, FactoryMethods)
{
    // Zeros
    TensorData<TypeParam> zeros = TensorData<TypeParam>::zeros({2, 3});
    EXPECT_EQ(zeros.shape()[0], 2);
    EXPECT_EQ(zeros.shape()[1], 3);
    for (const auto& val : zeros)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(0));
    }

    // Ones
    TensorData<TypeParam> ones = TensorData<TypeParam>::ones({3, 2});
    EXPECT_EQ(ones.shape()[0], 3);
    EXPECT_EQ(ones.shape()[1], 2);
    for (const auto& val : ones)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(1));
    }

    // Rand
    TensorData<TypeParam> random = TensorData<TypeParam>::rand({4, 4});
    EXPECT_EQ(random.shape()[0], 4);
    EXPECT_EQ(random.shape()[1], 4);
    for (const auto& val : random)
    {
        EXPECT_GE(val, static_cast<TypeParam>(0));
        EXPECT_LE(val, static_cast<TypeParam>(1));
    }
}

// Test scalar tensor operations
TYPED_TEST(TensorDataTest, ScalarTensorDataOperations)
{
    TensorData<TypeParam> scalar1({}, {5});
    TensorData<TypeParam> scalar2({}, {3});

    // Addition
    TensorData<TypeParam> sum = scalar1 + scalar2;
    EXPECT_EQ(sum.dim(), 0);
    EXPECT_EQ(sum.first(), static_cast<TypeParam>(8));

    // Multiplication
    TensorData<TypeParam> prod = scalar1 * scalar2;
    EXPECT_EQ(prod.dim(), 0);
    EXPECT_EQ(prod.first(), static_cast<TypeParam>(15));

    std::cout << "scalar1: " << &scalar1 << std::endl;
    std::cout << "scalar2: " << &scalar2 << std::endl;
    std::cout << "sum: " << &sum << std::endl;
    std::cout << "prod: " << &prod << std::endl;
}

TYPED_TEST(TensorDataTest, ReshapeDataIntegrity)
{
    TensorData<TypeParam> tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    tensor.reshape({4, 2});

    EXPECT_EQ(tensor(0, 0), static_cast<TypeParam>(1));
    EXPECT_EQ(tensor(0, 1), static_cast<TypeParam>(2));
    EXPECT_EQ(tensor(1, 0), static_cast<TypeParam>(3));
    EXPECT_EQ(tensor(1, 1), static_cast<TypeParam>(4));
    EXPECT_EQ(tensor(3, 1), static_cast<TypeParam>(8));
}

TYPED_TEST(TensorDataTest, MatmulEdgeCases)
{
    // Identity matrix
    TensorData<TypeParam> mat({2, 2}, {1, 0, 0, 1});
    TensorData<TypeParam> other({2, 2}, {5, 6, 7, 8});
    TensorData<TypeParam> result = mat.matmul(other);
    EXPECT_TRUE(result == other);

    // Non-commutative
    TensorData<TypeParam> mat1({2, 2}, {1, 2, 3, 4});
    TensorData<TypeParam> mat2({2, 2}, {5, 6, 7, 8});
    TensorData<TypeParam> result1 = mat1.matmul(mat2);
    TensorData<TypeParam> result2 = mat2.matmul(mat1);
    EXPECT_FALSE(result1 == result2);
    EXPECT_TRUE(result1 != result2);
}

// Test sum method
TYPED_TEST(TensorDataTest, SumMethod)
{
    // 1D TensorData
    TensorData<TypeParam> vec({4}, {1, 2, 3, 4});
    EXPECT_EQ(vec.sum(), static_cast<TypeParam>(10));

    // 2D TensorData
    TensorData<TypeParam> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(mat.sum(), static_cast<TypeParam>(21));

    // TensorData with single element
    TensorData<TypeParam> single({1}, {42});
    EXPECT_EQ(single.sum(), static_cast<TypeParam>(42));

    // TensorData with zeros
    TensorData<TypeParam> zeros = TensorData<TypeParam>::zeros({3, 3});
    EXPECT_EQ(zeros.sum(), static_cast<TypeParam>(0));
}

// Test broadcasting operations
TYPED_TEST(TensorDataTest, BroadcastingOperations)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(10));

    TensorData<TypeParam> scalar_tensor(static_cast<TypeParam>(5));

    // Test tensor + scalar_tensor
    auto result_add = tensor + scalar_tensor;
    for (const auto& val : result_add)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(15));
    }

    // Test scalar_tensor + tensor
    auto result_add_rev = scalar_tensor + tensor;
    for (const auto& val : result_add_rev)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(15));
    }

    // Test tensor - scalar_tensor
    auto result_sub = tensor - scalar_tensor;
    for (const auto& val : result_sub)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }

    // Test scalar_tensor - tensor
    auto result_sub_rev = scalar_tensor - tensor;
    for (const auto& val : result_sub_rev)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(-5));
    }

    // Test tensor * scalar_tensor
    auto result_mul = tensor * scalar_tensor;
    for (const auto& val : result_mul)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(50));
    }

    // Test scalar_tensor * tensor
    auto result_mul_rev = scalar_tensor * tensor;
    for (const auto& val : result_mul_rev)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(50));
    }

    // Test tensor / scalar_tensor
    auto result_div = tensor / scalar_tensor;
    for (const auto& val : result_div)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(2));
    }

    // Test scalar_tensor / tensor
    TensorData<TypeParam> tensor_for_div({2, 2});
    tensor_for_div.fill(static_cast<TypeParam>(2));
    TensorData<TypeParam> scalar_for_div(static_cast<TypeParam>(10));
    auto result_div_rev = scalar_for_div / tensor_for_div;
    for (const auto& val : result_div_rev)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test broadcasting with single-element, non-scalar tensor
TYPED_TEST(TensorDataTest, BroadcastingSingleElementTensorData)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(10));

    TensorData<TypeParam> single_element_tensor({1, 1},
                                            {static_cast<TypeParam>(5)});

    EXPECT_FALSE(single_element_tensor.isScalar());
    EXPECT_TRUE(single_element_tensor.hasOnlyOneVal());

    auto result_add = tensor + single_element_tensor;
    for (const auto& val : result_add)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(15));
    }

    auto result_mul = single_element_tensor * tensor;
    for (const auto& val : result_mul)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(50));
    }
}

// Test scalar multiplication commutativity
TYPED_TEST(TensorDataTest, ScalarMultiplicationCommutativity)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(3));
    TypeParam scalar = static_cast<TypeParam>(4);

    auto result1 = tensor * scalar;
    auto result2 = scalar * tensor;

    EXPECT_EQ(result1, result2);
}

// Test commutative scalar operations
TYPED_TEST(TensorDataTest, CommutativeScalarOperations)
{
    TensorData<TypeParam> tensor({2, 2});
    tensor.fill(static_cast<TypeParam>(10));
    TypeParam scalar = static_cast<TypeParam>(5);

    // Test scalar + tensor
    auto result_add = scalar + tensor;
    for (const auto& val : result_add)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(15));
    }

    // Test scalar - tensor
    auto result_sub = scalar - tensor;
    for (const auto& val : result_sub)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(-5));
    }

    // Test scalar / tensor
    TensorData<TypeParam> tensor_for_div({2, 2});
    tensor_for_div.fill(static_cast<TypeParam>(2));
    TypeParam scalar_for_div = static_cast<TypeParam>(10);
    auto result_div = scalar_for_div / tensor_for_div;
    for (const auto& val : result_div)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(5));
    }
}

// Test operator[] access
TYPED_TEST(TensorDataTest, OperatorBracketAccess)
{
    TensorData<TypeParam> tensor({2, 3}, {1, 2, 3, 4, 5, 6});

    // Read access
    EXPECT_EQ(tensor[0], static_cast<TypeParam>(1));
    EXPECT_EQ(tensor[5], static_cast<TypeParam>(6));

    // Write access
    tensor[1] = static_cast<TypeParam>(22);
    EXPECT_EQ(tensor[1], static_cast<TypeParam>(22));
}

// Test tensor slicing with at() method
TEST_F(TensorDataFloatTest, TensorDataSlicing3D)
{
    TensorData<f32> tensor({2, 3, 4});
    tensor.fill(5.0f);

    // Get a 2D slice by fixing first dimension
    auto result = tensor.at({0});
    EXPECT_EQ(result.dim(), 2);
    EXPECT_EQ(result.size(), 12); // 3 * 4

    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 5.0f);
    }
}

// Test tensor slicing with at() method - 2D to 1D
TEST_F(TensorDataFloatTest, TensorDataSlicing2DTo1D)
{
    TensorData<f32> tensor({3, 4});

    // Fill with sequential values
    for (sizeT i = 0; i < 3; ++i)
    {
        for (sizeT j = 0; j < 4; ++j)
        {
            tensor.set({i, j}, static_cast<f32>(i * 4 + j));
        }
    }

    // Get the second row
    auto result = tensor.at({1});
    EXPECT_EQ(result.dim(), 1);
    EXPECT_EQ(result.size(), 4);

    EXPECT_FLOAT_EQ(result.at({0}).first(), 4.0f);
    EXPECT_FLOAT_EQ(result.at({1}).first(), 5.0f);
    EXPECT_FLOAT_EQ(result.at({2}).first(), 6.0f);
    EXPECT_FLOAT_EQ(result.at({3}).first(), 7.0f);
}

// Test tensor slicing with at() method - 2D to 0D (scalar)
TEST_F(TensorDataFloatTest, TensorDataSlicing2DTo0D)
{
    TensorData<f32> tensor({3, 4});
    tensor.fill(0.0f);
    tensor.set({1, 2}, 42.0f);

    auto scalar_tensor = tensor.at({1, 2});
    EXPECT_EQ(scalar_tensor.dim(), 0);
    EXPECT_EQ(scalar_tensor.size(), 1);
    EXPECT_FLOAT_EQ(scalar_tensor.first(), 42.0f);
}

// Test at() with wrong number of indices
TEST_F(TensorDataFloatTest, AtMethodTooManyIndices)
{
    TensorData<f32> tensor({2, 3});
    EXPECT_THROW((void) tensor.at({0, 1, 2}), IndexOutOfBoundError);
}

// Test at() with empty indices on non-empty tensor
TEST_F(TensorDataFloatTest, AtMethodEmptyIndices)
{
    TensorData<f32> tensor({2, 3});
    EXPECT_THROW((void) tensor.at({}), IndexOutOfBoundError);
}

// Test set() with wrong number of indices
TEST_F(TensorDataFloatTest, SetMethodTooManyIndices)
{
    TensorData<f32> tensor({2, 3});
    EXPECT_THROW(tensor.set({0, 1, 2}, 5.0f), IndexOutOfBoundError);
}

// Test set() with empty indices on non-empty tensor
TEST_F(TensorDataFloatTest, SetMethodEmptyIndices)
{
    TensorData<f32> tensor({2, 3});
    EXPECT_THROW(tensor.set({}, 5.0f), IndexOutOfBoundError);
}

// Test copy with Vector size mismatch
TEST_F(TensorDataFloatTest, CopyFromVecSizeMismatch)
{
    ds::Vector<f32> vec;
    vec.pushBack(1.0f);
    vec.pushBack(2.0f);
    vec.pushBack(3.0f);

    TensorData<f32> tensor({2, 2}); // Size 4, but vec has size 3
    EXPECT_THROW(tensor.copy(vec), TensorDataErr);
}

// Test TensorDataErr is returned for invalid copy
TEST_F(TensorDataFloatTest, InvalidCopyReturnsTensorDataErr)
{
    TensorData<f32> source({2, 3});
    TensorData<f32> dest({3, 2});

    EXPECT_THROW(dest.copy(source), TensorDataErr);
}

// Test checkShape logic with same size, different shape
TYPED_TEST(TensorDataTest, CheckShapeSameSizeDifferentShape)
{
    TensorData<TypeParam> tensor1({2, 6});
    TensorData<TypeParam> tensor2({3, 4});

    tensor1.fill(static_cast<TypeParam>(2));
    tensor2.fill(static_cast<TypeParam>(3));

    // Redirect cout to a stringstream to capture the warning
    std::stringstream buffer;
    std::streambuf* old_cout = std::cout.rdbuf(buffer.rdbuf());

    // This should not throw, but should print a warning
    TensorData<TypeParam> result = tensor1 * tensor2;

    // Restore cout
    std::cout.rdbuf(old_cout);

    std::string output = buffer.str();
    EXPECT_NE(
        output.find("warn: same size but different shape tensors multiply"),
        std::string::npos);

    // Verify the result is as expected (element-wise multiplication)
    for (const auto& val : result)
    {
        EXPECT_EQ(val, static_cast<TypeParam>(6));
    }
}

// Test computeSize with 0D tensor
TEST_F(TensorDataFloatTest, ZeroDimensionalTensorData)
{
    TensorData<f32> scalar({1});
    EXPECT_EQ(scalar.dim(), 1);
    EXPECT_EQ(scalar.size(), 1);

    scalar.fill(42.0f);
    EXPECT_FLOAT_EQ(scalar.first(), 42.0f);
}

// Test large tensor operations
TEST_F(TensorDataFloatTest, LargeTensorDataOperations)
{
    TensorData<f32> tensor1({10, 10, 10});
    TensorData<f32> tensor2({10, 10, 10});

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
TEST_F(TensorDataFloatTest, MixedOperations)
{
    TensorData<f32> a({3, 3});
    TensorData<f32> b({3, 3});

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
TEST_F(TensorDataFloatTest, CompoundOperationsPreserveShape)
{
    TensorData<f32> tensor1({2, 3, 4});
    TensorData<f32> tensor2({2, 3, 4});

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
TEST_F(TensorDataFloatTest, DotProductWithZeros)
{
    TensorData<f32> tensor1({5});
    TensorData<f32> tensor2({5});

    tensor1.fill(5.0f);
    tensor2.fill(0.0f);

    auto result = tensor1.dot(tensor2);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// Test dot product commutative property
TEST_F(TensorDataFloatTest, DotProductCommutative)
{
    TensorData<f32> tensor1({4});
    TensorData<f32> tensor2({4});

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
TEST_F(TensorDataFloatTest, IndexCalculation4D)
{
    TensorData<f32> tensor({2, 3, 4, 5});

    // Test corner indices
    EXPECT_EQ(tensor.index({0, 0, 0, 0}), 0);
    // (1*3*4*5) + (2*4*5) + (3*5) + 4 = 60 + 40 + 15 + 4 = 119
    EXPECT_EQ(tensor.index({1, 2, 3, 4}), 119);
}

// Test rawData returns correct data
TEST_F(TensorDataFloatTest, RawDataReturnsCorrectValues)
{
    TensorData<f32> tensor({2, 3});

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
TEST_F(TensorDataFloatTest, FromVectorLarge)
{
    ds::Vector<f32> vec;
    for (sizeT i = 0; i < 100; ++i)
    {
        vec.pushBack(static_cast<f32>(i));
    }

    auto tensor = TensorData<f32>::fromVector(vec);

    EXPECT_EQ(tensor.size(), 100);
    EXPECT_EQ(tensor.dim(), 1);

    for (sizeT i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(tensor.at({i}).first(), static_cast<f32>(i));
    }
}

// Test negative values
TEST_F(TensorDataFloatTest, NegativeValues)
{
    TensorData<f32> tensor({3, 3});
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
TEST_F(TensorDataFloatTest, VerySmallValues)
{
    TensorData<f32> tensor({2, 2});
    tensor.fill(1e-10f);

    auto result = tensor + tensor;
    for (const auto& val : result)
    {
        EXPECT_FLOAT_EQ(val, 2e-10f);
    }
}

// Integer tensor specific tests
class TensorDataIntTest : public ::testing::Test
{
};

TEST_F(TensorDataIntTest, IntegerDivision)
{
    TensorData<int> tensor1({2, 2});
    TensorData<int> tensor2({2, 2});

    tensor1.fill(10);
    tensor2.fill(3);

    auto result = tensor1 / tensor2;

    for (const auto& val : result)
    {
        EXPECT_EQ(val, 3); // Integer division
    }
}

TEST_F(TensorDataIntTest, IntegerArithmetic)
{
    TensorData<int> tensor({3});
    tensor.set({0}, 100);
    tensor.set({1}, 200);
    tensor.set({2}, 300);

    auto result = (tensor + 50) * 2 - 100;

    EXPECT_EQ(result.at({0}).first(), 200); // (100+50)*2-100 = 200
    EXPECT_EQ(result.at({1}).first(), 400); // (200+50)*2-100 = 400
    EXPECT_EQ(result.at({2}).first(), 600); // (300+50)*2-100 = 600
}
