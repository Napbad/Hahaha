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

#include <cstdlib>
#include <gtest/gtest.h>

#include "backend/Device.h" // Include for DeviceType
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h" // Include for TensorShape

class TensorWrapperTest : public ::testing::Test {
protected:
    // Common setup/teardown if needed
};

using hahaha::math::TensorWrapper;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::backend::Device;
using hahaha::backend::DeviceType;

// --- Constructor Tests ---

TEST_F(TensorWrapperTest, Constructor_Default_CreatesEmptyTensor) {
    TensorWrapper<int> tensor;
    EXPECT_EQ(tensor.getTotalSize(), 0);
    EXPECT_EQ(tensor.getShape().size(), 0);
}

TEST_F(TensorWrapperTest, Constructor_ShapeInitValueDevice_CreatesCorrectTensor) {
    TensorWrapper<float> tensor(TensorShape({2, 3}), 1.0f, Device(DeviceType::CPU, 0));
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.at({0, 0}), 1.0f);
    EXPECT_EQ(tensor.at({1, 2}), 1.0f);
}

TEST_F(TensorWrapperTest, Constructor_ShapeDevice_CreatesCorrectTensorWithDefaultInit) {
    TensorWrapper<int> tensor(TensorShape({2, 2}), Device(DeviceType::CPU, 0));
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.at({0, 0}), 0); // Default init value for int
}

TEST_F(TensorWrapperTest, Constructor_NestedData_2D_CorrectlyInitializes) {
    TensorWrapper<int> tensor(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.at({0, 0}), 1);
    EXPECT_EQ(tensor.at({1, 1}), 4);
}

TEST_F(TensorWrapperTest, Constructor_NestedData_3D_CorrectlyInitializes) {
    TensorWrapper<int> tensor(NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(tensor.getTotalSize(), 8);
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.at({0, 0, 0}), 1);
    EXPECT_EQ(tensor.at({1, 1, 1}), 8);
}

TEST_F(TensorWrapperTest, Constructor_NestedData_SingleValue_CorrectlyInitializes) {
    TensorWrapper<float> tensor(42.0f);
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 0); // Scalar tensor has 0 dimensions
    EXPECT_EQ(tensor.at({}), 42.0f);
}

TEST_F(TensorWrapperTest, Constructor_NestedData_IrregularShape_ThrowsInvalidArgument) {
    EXPECT_THROW(TensorWrapper<int>(NestedData<int>{{1}, {1, 2}}), std::invalid_argument);
    EXPECT_THROW(TensorWrapper<float>(NestedData<float>{{1.0f, 2.0f}, {3.0f}}), std::invalid_argument);
}

TEST_F(TensorWrapperTest, Constructor_Vector_CorrectlyInitializes) {
    std::vector<int> vec = {1, 2, 3};
    TensorWrapper<int> tensor(vec);
    EXPECT_EQ(tensor.getTotalSize(), 3);
    EXPECT_EQ(tensor.getShape().size(), 1);
    EXPECT_EQ(tensor.at({0}), 1);
    EXPECT_EQ(tensor.at({2}), 3);
}

// --- Property and Accessor Tests ---

TEST_F(TensorWrapperTest, GetShape_ReturnsCorrectDimensionsAndSize) {
    TensorWrapper<int> tensor_2d(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_2d.getShape()[0], 2);
    EXPECT_EQ(tensor_2d.getShape()[1], 2);
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);

    TensorWrapper<int> tensor_3d(NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(tensor_3d.getShape()[0], 2);
    EXPECT_EQ(tensor_3d.getShape()[1], 2);
    EXPECT_EQ(tensor_3d.getShape()[2], 2);
    EXPECT_EQ(tensor_3d.getTotalSize(), 8);
}

TEST_F(TensorWrapperTest, GetStride_ReturnsCorrectValues) {
    TensorWrapper<int> tensor_2d(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_2d.getStride().getDims()[0], 2);
    EXPECT_EQ(tensor_2d.getStride().getDims()[1], 1);

    TensorWrapper<int> tensor_3d(NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(tensor_3d.getStride().getDims()[0], 4);
    EXPECT_EQ(tensor_3d.getStride().getDims()[1], 2);
    EXPECT_EQ(tensor_3d.getStride().getDims()[2], 1);
}

TEST_F(TensorWrapperTest, GetDevice_ReturnsCorrectDefaultDevice) {
    TensorWrapper<int> tensor(TensorShape({2, 2}));
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.getDevice().id, 0);
}

TEST_F(TensorWrapperTest, ElementAccess_ReadWrite_CorrectlyModifiesAndRetrieves) {
    TensorWrapper<int> tensor(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor.at({0, 0}), 1);
    tensor.at({1, 1}) = 10;
    EXPECT_EQ(tensor.at({1, 1}), 10);
}

TEST_F(TensorWrapperTest, ElementAccess_OutOfBounds_ThrowsOutOfRange) {
    TensorWrapper<int> tensor(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_THROW(tensor.at({0, 0, 0}), std::out_of_range); // Dimension mismatch
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range);     // Index out of bounds
}

TEST_F(TensorWrapperTest, GetDimensions_ReturnsCorrectCount) {
    TensorWrapper<int> tensor_scalar(1);
    EXPECT_EQ(tensor_scalar.getDimensions(), 0);
    TensorWrapper<int> tensor_1d(NestedData<int>{1, 2, 3});
    EXPECT_EQ(tensor_1d.getDimensions(), 1);
    TensorWrapper<int> tensor_2d(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_2d.getDimensions(), 2);
}

TEST_F(TensorWrapperTest, GetTotalSize_ReturnsCorrectCount) {
    TensorWrapper<int> tensor_scalar(NestedData<int>{1});
    EXPECT_EQ(tensor_scalar.getTotalSize(), 1);
    TensorWrapper<int> tensor_1d(NestedData<int>{1, 2, 3});
    EXPECT_EQ(tensor_1d.getTotalSize(), 3);
    TensorWrapper<int> tensor_2d(NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);
}

// --- Move/Copy Semantics Tests ---

TEST_F(TensorWrapperTest, MoveConstructor_TransfersOwnershipAndLeavesSourceValidButUnspecified) {
    TensorWrapper<int> tensor_orig(NestedData<int>{{1, 2}, {3, 4}});
    TensorWrapper<int> tensor_moved(std::move(tensor_orig));
    EXPECT_EQ(tensor_moved.getTotalSize(), 4);
    // tensor_orig's state is valid but unspecified, can check for some properties
    EXPECT_EQ(tensor_orig.getTotalSize(), 0); // Moved-from objects usually have empty state
}

TEST_F(TensorWrapperTest, MoveAssignment_TransfersOwnership) {
    TensorWrapper<int> tensor_a(NestedData<int>{{1, 1}});
    TensorWrapper<int> tensor_b(NestedData<int>{{2, 2}});
    tensor_a = std::move(tensor_b);
    EXPECT_EQ(tensor_a.at({0, 0}), 2);
    EXPECT_EQ(tensor_b.getTotalSize(), 0); // Moved-from object is empty
}

TEST_F(TensorWrapperTest, CopyConstructor_DeepCopiesData) {
    TensorWrapper<int> original(NestedData<int>{{1, 2}});
    TensorWrapper<int> copy = original; // Uses copy constructor
    original.at({0, 0}) = 10;
    EXPECT_EQ(copy.at({0, 0}), 1); // Original modification should not affect copy
}

// Note: Copy assignment operator is explicitly deleted, so no test is needed for it to fail.

// --- Binary Operations (Tensor op Tensor) ---

TEST_F(TensorWrapperTest, Add_TwoTensors_CorrectResult) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> t2(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    auto res = t1 + t2;
    EXPECT_EQ(res.at({0, 0}), 6.0f);
    EXPECT_EQ(res.at({1, 1}), 12.0f);
}

TEST_F(TensorWrapperTest, Add_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t2(NestedData<float>{{1.0f, 2.0f, 3.0f}});
    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

// TEST_F(TensorWrapperTest, Add_TwoTensors_DeviceMismatch_ThrowsInvalidArgument) {
//     // Assuming a way to create non-CPU device for testing
//     // For now, this will only test if the check is present.
//     TensorWrapper<float> t1({2, 2}, 1.0f, Device(DeviceType::CPU, 0));
//     TensorWrapper<float> t2({2, 2}, 1.0f, Device(DeviceType::GPU, 0));
//     EXPECT_THROW(t1 + t2, std::invalid_argument);
// }

TEST_F(TensorWrapperTest, Subtract_TwoTensors_CorrectResult) {
    TensorWrapper<float> t1(NestedData<float>{{5.0f, 6.0f}});
    TensorWrapper<float> t2(NestedData<float>{{1.0f, 2.0f}});
    auto res = t1 - t2;
    EXPECT_EQ(res.at({0, 0}), 4.0f);
    EXPECT_EQ(res.at({0, 1}), 4.0f);
}

TEST_F(TensorWrapperTest, Subtract_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t2(NestedData<float>{{1.0f, 2.0f, 3.0f}});
    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

TEST_F(TensorWrapperTest, Multiply_TwoTensors_CorrectResult) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t2(NestedData<float>{{3.0f, 4.0f}});
    auto res = t1 * t2;
    EXPECT_EQ(res.at({0, 0}), 3.0f);
    EXPECT_EQ(res.at({0, 1}), 8.0f);
}

TEST_F(TensorWrapperTest, Multiply_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t2(NestedData<float>{{1.0f, 2.0f, 3.0f}});
    EXPECT_THROW(t1 * t2, std::invalid_argument);
}

TEST_F(TensorWrapperTest, Divide_TwoTensors_CorrectResult) {
    TensorWrapper<float> t1(NestedData<float>{{10.0f, 20.0f}});
    TensorWrapper<float> t2(NestedData<float>{{2.0f, 4.0f}});
    auto res = t1 / t2;
    EXPECT_EQ(res.at({0, 0}), 5.0f);
    EXPECT_EQ(res.at({0, 1}), 5.0f);
}

TEST_F(TensorWrapperTest, Divide_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t2(NestedData<float>{{1.0f, 2.0f, 3.0f}});
    EXPECT_THROW(t1 / t2, std::invalid_argument);
}

TEST_F(TensorWrapperTest, Divide_TwoTensors_DivisionByZero_ThrowsRuntimeError) {
    TensorWrapper<float> t1(NestedData<float>{{1.0f, 2.0f}});
    TensorWrapper<float> t_zero(NestedData<float>{{0.0f, 1.0f}});
    EXPECT_THROW(t1 / t_zero, std::runtime_error);
}

// --- Scalar Operations (Tensor op Scalar, Scalar op Tensor) ---

TEST_F(TensorWrapperTest, Add_TensorScalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    float scalar = 10.0f;
    auto res = tensor + scalar;
    EXPECT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_EQ(res.at({0, 1}), 12.0f);
}

TEST_F(TensorWrapperTest, Add_ScalarTensor_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    float scalar = 10.0f;
    auto res = scalar + tensor;
    EXPECT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_EQ(res.at({0, 1}), 12.0f);
}

TEST_F(TensorWrapperTest, Subtract_TensorScalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{10.0f, 20.0f}});
    float scalar = 5.0f;
    auto res = tensor - scalar;
    EXPECT_EQ(res.at({0, 0}), 5.0f);
    EXPECT_EQ(res.at({0, 1}), 15.0f);
}

TEST_F(TensorWrapperTest, Subtract_ScalarTensor_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{10.0f, 20.0f}});
    float scalar = 30.0f;
    auto res = scalar - tensor; // 30 - {10, 20} = {20, 10}
    EXPECT_EQ(res.at({0, 0}), 20.0f);
    EXPECT_EQ(res.at({0, 1}), 10.0f);
}

TEST_F(TensorWrapperTest, Multiply_TensorScalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    float scalar = 5.0f;
    auto res = tensor * scalar;
    EXPECT_EQ(res.at({0, 0}), 5.0f);
    EXPECT_EQ(res.at({0, 1}), 10.0f);
}

TEST_F(TensorWrapperTest, Multiply_ScalarTensor_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    float scalar = 5.0f;
    auto res = scalar * tensor;
    EXPECT_EQ(res.at({0, 0}), 5.0f);
    EXPECT_EQ(res.at({0, 1}), 10.0f);
}

TEST_F(TensorWrapperTest, Divide_TensorScalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{10.0f, 20.0f}});
    float scalar = 2.0f;
    auto res = tensor / scalar;
    EXPECT_EQ(res.at({0, 0}), 5.0f);
    EXPECT_EQ(res.at({0, 1}), 10.0f);
}

TEST_F(TensorWrapperTest, Divide_TensorScalar_DivisionByZero_ThrowsRuntimeError) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    EXPECT_THROW(tensor / 0.0f, std::runtime_error);
}

TEST_F(TensorWrapperTest, Divide_ScalarTensor_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{10.0f, 20.0f}});
    float scalar = 100.0f;
    auto res = scalar / tensor; // 100 / {10, 20} = {10, 5}
    EXPECT_EQ(res.at({0, 0}), 10.0f);
    EXPECT_EQ(res.at({0, 1}), 5.0f);
}

TEST_F(TensorWrapperTest, Divide_ScalarTensor_DivisionByZero_ThrowsRuntimeError) {
    TensorWrapper<float> tensor_with_zero(NestedData<float>{{0.0f, 1.0f}});
    EXPECT_THROW(10.0f / tensor_with_zero, std::runtime_error);
}

// --- In-Place Operations ---

TEST_F(TensorWrapperTest, InPlaceAdd_TwoTensors_CorrectResult) {
    TensorWrapper<int> t1(NestedData<int>{{1, 2}, {3, 4}});
    TensorWrapper<int> t2(NestedData<int>{{5, 6}, {7, 8}});
    t1 += t2;
    EXPECT_EQ(t1.at({0, 0}), 6);
    EXPECT_EQ(t1.at({1, 1}), 12);
}

TEST_F(TensorWrapperTest, InPlaceAdd_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<int> t1(NestedData<int>{{1, 2}});
    TensorWrapper<int> t2(NestedData<int>{1, 2, 3});
    EXPECT_THROW(t1 += t2, std::invalid_argument);
}

TEST_F(TensorWrapperTest, InPlaceAdd_Scalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    tensor += 2.0f;
    EXPECT_EQ(tensor.at({0, 0}), 3.0f);
    EXPECT_EQ(tensor.at({0, 1}), 4.0f);
}

TEST_F(TensorWrapperTest, InPlaceSubtract_Scalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{5.0f, 6.0f}});
    tensor -= 1.0f;
    EXPECT_EQ(tensor.at({0, 0}), 4.0f);
    EXPECT_EQ(tensor.at({0, 1}), 5.0f);
}

TEST_F(TensorWrapperTest, InPlaceMultiply_Scalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    tensor *= 3.0f;
    EXPECT_EQ(tensor.at({0, 0}), 3.0f);
    EXPECT_EQ(tensor.at({0, 1}), 6.0f);
}

TEST_F(TensorWrapperTest, InPlaceDivide_Scalar_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{10.0f, 20.0f}});
    tensor /= 5.0f;
    EXPECT_EQ(tensor.at({0, 0}), 2.0f);
    EXPECT_EQ(tensor.at({0, 1}), 4.0f);
}

TEST_F(TensorWrapperTest, InPlaceDivide_Scalar_DivisionByZero_ThrowsRuntimeError) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}});
    EXPECT_THROW(tensor /= 0.0f, std::runtime_error);
}

// --- Unary Operations ---

TEST_F(TensorWrapperTest, UnaryNegation_CorrectResult) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, -2.0f}});
    auto neg_tensor = -tensor;
    EXPECT_EQ(neg_tensor.at({0, 0}), -1.0f);
    EXPECT_EQ(neg_tensor.at({0, 1}), 2.0f);
}

TEST_F(TensorWrapperTest, Sum_CorrectResultForFloatTensor) {
    TensorWrapper<float> tensor(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_EQ(tensor.sum(), 10.0f);
}

TEST_F(TensorWrapperTest, Sum_CorrectResultForIntTensor) {
    TensorWrapper<int> tensor(NestedData<int>{1, 2, 3, 4, 5});
    EXPECT_EQ(tensor.sum(), 15);
}

// --- Transformation Operations ---

TEST_F(TensorWrapperTest, Reshape_ValidNewShape_CorrectResult) {
    TensorWrapper<int> tensor_orig(NestedData<int>{1, 2, 3, 4, 5, 6});
    auto tensor_reshaped = tensor_orig.reshape({2, 3});
    EXPECT_EQ(tensor_reshaped.getShape().size(), 2);
    EXPECT_EQ(tensor_reshaped.at({0, 0}), 1);
    EXPECT_EQ(tensor_reshaped.at({1, 2}), 6);
}

TEST_F(TensorWrapperTest, Reshape_SizeMismatch_ThrowsInvalidArgument) {
    TensorWrapper<int> tensor_orig(NestedData<int>{1, 2, 3, 4});
    EXPECT_THROW(tensor_orig.reshape({3}), std::invalid_argument);   // Total size 3 != 4
    EXPECT_THROW(tensor_orig.reshape({2, 3}), std::invalid_argument); // Total size 6 != 4
}

TEST_F(TensorWrapperTest, Matmul_ValidMatrices_CorrectResult) {
    TensorWrapper<float> matrix_a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    TensorWrapper<float> matrix_b(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}}); // 2x2
    auto matrix_c = matrix_a.matmul(matrix_b);
    EXPECT_EQ(matrix_c.at({0, 0}), 19.0f); // 1*5 + 2*7 = 5 + 14 = 19
    EXPECT_EQ(matrix_c.at({0, 1}), 22.0f); // 1*6 + 2*8 = 6 + 16 = 22
    EXPECT_EQ(matrix_c.at({1, 0}), 43.0f); // 3*5 + 4*7 = 15 + 28 = 43
    EXPECT_EQ(matrix_c.at({1, 1}), 50.0f); // 3*6 + 4*8 = 18 + 32 = 50
}

TEST_F(TensorWrapperTest, Matmul_Non2DTensors_ThrowsInvalidArgument) {
    TensorWrapper<float> matrix_a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    TensorWrapper<float> tensor_1d(NestedData<float>{1.0f, 2.0f});               // 1D
    EXPECT_THROW(matrix_a.matmul(tensor_1d), std::invalid_argument);
    EXPECT_THROW(tensor_1d.matmul(matrix_a), std::invalid_argument);
}

TEST_F(TensorWrapperTest, Matmul_InnerDimensionMismatch_ThrowsInvalidArgument) {
    TensorWrapper<float> matrix_a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});               // 2x2
    TensorWrapper<float> matrix_d(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}); // 3x2
    EXPECT_THROW(matrix_a.matmul(matrix_d), std::invalid_argument); // (2x2) @ (3x2) -> inner dims 2 != 3
}

TEST_F(TensorWrapperTest, Transpose_Valid2DTensor_CorrectResult) {
    TensorWrapper<int> tensor_orig(NestedData<int>{{1, 2, 3}, {4, 5, 6}}); // 2x3
    auto tensor_transposed = tensor_orig.transpose();                 // 3x2
    EXPECT_EQ(tensor_transposed.getShape()[0], 3);
    EXPECT_EQ(tensor_transposed.getShape()[1], 2);
    EXPECT_EQ(tensor_transposed.at({0, 0}), 1);
    EXPECT_EQ(tensor_transposed.at({0, 1}), 4);
    EXPECT_EQ(tensor_transposed.at({2, 1}), 6);
}

TEST_F(TensorWrapperTest, Transpose_Non2DTensor_ThrowsInvalidArgument) {
    TensorWrapper<int> tensor_1d(NestedData<int>{1, 2, 3}); // 1D
    EXPECT_THROW(tensor_1d.transpose(), std::invalid_argument);
}

// --- Device and Special Operations ---

TEST_F(TensorWrapperTest, Clear_ResetsTensorDataToZero) {
    TensorWrapper<int> tensor({2, 2}, 5); // All elements are 5
    tensor.clear();
    EXPECT_EQ(tensor.at({0, 0}), 0);
    EXPECT_EQ(tensor.at({1, 1}), 0);
}

TEST_F(TensorWrapperTest, To_SameDevice_NoChangeAndNoThrow) {
    TensorWrapper<float> tensor({2, 2}, 1.0f, Device(DeviceType::CPU, 0));
    // Should not throw, device remains the same
    EXPECT_NO_THROW(tensor.to(Device(DeviceType::CPU, 0)));
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
}

TEST_F(TensorWrapperTest, To_DifferentDevice_NotImplemented_ThrowsRuntimeError) {
    TensorWrapper<float> tensor({2, 2}, 1.0f, Device(DeviceType::CPU, 0));
    EXPECT_THROW(tensor.to(Device(DeviceType::GPU, 0)), std::runtime_error);
}

TEST_F(TensorWrapperTest, Broadcast_NotImplemented_ThrowsRuntimeError) {
    TensorWrapper<int> tensor1(NestedData<int>{1});
    TensorWrapper<int> tensor2(NestedData<int>{1, 2});
    EXPECT_THROW(tensor1.broadcast(tensor2), std::runtime_error);
}
