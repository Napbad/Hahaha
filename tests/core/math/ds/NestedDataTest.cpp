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
// 
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
// 

#include <gtest/gtest.h>

#include "math/ds/NestedData.h"

using hahaha::math::NestedData;

class NestedDataTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }

};

TEST_F(NestedDataTest, InitializeViaInitializerList) {
    NestedData<int> nd({1, 2, 3, 4, 5});
    ASSERT_EQ(nd.getFlatData().size(), 5);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 5);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
}

TEST_F(NestedDataTest, InitializeViaNestedInitializerList1) {
    NestedData<int> nd = {
        NestedData<int>({1, 2, 3}),
        NestedData<int>({4, 5, 6}),
        NestedData<int>({7, 8, 9}),
    };
    ASSERT_EQ(nd.getFlatData().size(), 9);
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape().at(0), 3);
    ASSERT_EQ(nd.getShape().at(1), 3);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
    ASSERT_EQ(nd.getFlatData().at(5), 6);
    ASSERT_EQ(nd.getFlatData().at(6), 7);
    ASSERT_EQ(nd.getFlatData().at(7), 8);
    ASSERT_EQ(nd.getFlatData().at(8), 9);
}

TEST_F(NestedDataTest, InitializeViaNestedInitializerList2) {
    NestedData<int> nd = {
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        }, 
    };
    ASSERT_EQ(nd.getFlatData().size(), 8);
    ASSERT_EQ(nd.getShape().size(), 3);
    ASSERT_EQ(nd.getShape().at(0), 2);
    ASSERT_EQ(nd.getShape().at(1), 2);
    ASSERT_EQ(nd.getShape().at(2), 2);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
    ASSERT_EQ(nd.getFlatData().at(5), 6);
    ASSERT_EQ(nd.getFlatData().at(6), 7);
    ASSERT_EQ(nd.getFlatData().at(7), 8);
}


TEST_F(NestedDataTest, InitializeViaNestedInitializerList3) {
    NestedData<int> nestedData({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });


    ASSERT_EQ(nestedData.getFlatData().size(), 8);
    ASSERT_EQ(nestedData.getShape().size(), 3);
    ASSERT_EQ(nestedData.getShape().at(0), 2);
    ASSERT_EQ(nestedData.getShape().at(1), 2);
    ASSERT_EQ(nestedData.getShape().at(2), 2);
    ASSERT_EQ(nestedData.getFlatData().at(0), 1);
    ASSERT_EQ(nestedData.getFlatData().at(1), 2);
    ASSERT_EQ(nestedData.getFlatData().at(2), 3);
    ASSERT_EQ(nestedData.getFlatData().at(3), 4);
    ASSERT_EQ(nestedData.getFlatData().at(4), 5);
    ASSERT_EQ(nestedData.getFlatData().at(5), 6);
    ASSERT_EQ(nestedData.getFlatData().at(6), 7);
    ASSERT_EQ(nestedData.getFlatData().at(7), 8);
}

TEST_F(NestedDataTest, InitializeWithEmptyList) {
    ASSERT_NO_THROW(NestedData<int> nd({}));
    NestedData<int> nd({});
    ASSERT_EQ(nd.getFlatData().size(), 0);
    ASSERT_EQ(nd.getShape().size(), 0);
}

TEST_F(NestedDataTest, SingleValueConstruction) {
    NestedData<int> nd(42);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getFlatData()[0], 42);
    ASSERT_EQ(nd.getShape().size(), 0);
}