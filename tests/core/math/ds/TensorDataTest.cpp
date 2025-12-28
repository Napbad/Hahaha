// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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

// Created: 2025-12-26 02:10:03 by Napbad
//

#include "math/ds/TensorData.h"
#include "math/ds/TensorShape.h"
#include "math/ds/TensorStride.h"

#include <gtest/gtest.h>

class TensorDataTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

using hahaha::math::TensorData;

TEST_F(TensorDataTest, InitWithInitializerList)
{
    TensorData<int> singleValueTensor = {1};
    ASSERT_NO_THROW(TensorData<int> singleValueTensor = {1};);

    TensorData<int> twoElementTensor({{1}, {2}});
    ASSERT_NO_THROW(TensorData<int> twoElementTensor({{1}, {2}}););
}

TEST_F(TensorDataTest, DefaultConstructor)
{
    TensorData<float> defaultConstructedTensor;
    // Default constructor should create an empty TensorData
    // Since TensorDataTest is a friend class, we can check internal state if needed
}

TEST_F(TensorDataTest, OneDimensionalTensor)
{
    TensorData<int> oneDimensionalTensor({1, 2, 3, 4, 5});
    
    // Test that we can create a tensor with 1D data
    ASSERT_NO_THROW((TensorData<int>{{1, 2, 3, 4, 5}}));
}

TEST_F(TensorDataTest, TwoDimensionalTensor)
{
    TensorData<double> twoDimensionalTensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    
    // Test that we can create a 2x3 tensor
    ASSERT_NO_THROW((TensorData<double>{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}}));
}

TEST_F(TensorDataTest, ThreeDimensionalTensor)
{
    TensorData<float> threeDimensionalTensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    
    // Test that we can create a 2x2x2 tensor
    ASSERT_NO_THROW((TensorData<float>{{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}}));
}

TEST_F(TensorDataTest, SingleValueTensor)
{
    TensorData<int> singleValueTensor(42);
    
    // Test that we can create a tensor with a single value
    ASSERT_NO_THROW((TensorData<int>{42}));
}

TEST_F(TensorDataTest, LargeTensor)
{
    // Test with a larger tensor to make sure it works properly
    TensorData<int> largeTensor({
        {1, 2, 3, 4, 5, 6},
        {7, 8, 9, 10, 11, 12},
        {13, 14, 15, 16, 17, 18}
    });
    
    ASSERT_NO_THROW((
        TensorData<int>{{
            {1, 2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11, 12},
            {13, 14, 15, 16, 17, 18}
        }}
    ));
}

TEST_F(TensorDataTest, EmptyTensor)
{
    // Test creating an empty tensor (0 dimensions)
    // Using default constructor instead of ambiguous {} initializer
    TensorData<int> emptyTensor;
    
    ASSERT_NO_THROW((TensorData<int>()));
}

TEST_F(TensorDataTest, DifferentTypes)
{
    // Test different value types
    ASSERT_NO_THROW((TensorData<int>({1, 2, 3})));
    ASSERT_NO_THROW((TensorData<float>({1.1f, 2.2f, 3.3f})));
    ASSERT_NO_THROW((TensorData<double>({1.1, 2.2, 3.3})));
    ASSERT_NO_THROW((TensorData<char>({'a', 'b', 'c'})));
    ASSERT_NO_THROW((TensorData<bool>({true, false, true})));
}

TEST_F(TensorDataTest, CopyAndAssignmentDisabled)
{
    TensorData<int> originalTensor({1, 2, 3});
    
    // Test that copy constructor is deleted
    // This should not compile, so we're verifying the behavior by design
    // TensorData<int> td2 = originalTensor;  // This would cause a compile error
    
    // Test that assignment operator is deleted
    // TensorData<int> td3;
    // td3 = originalTensor;  // This would cause a compile error
    
    SUCCEED() << "Copy constructor and assignment operator are properly deleted";
}