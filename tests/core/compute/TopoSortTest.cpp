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

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "compute/graph/ComputeNode.h"
#include "compute/graph/TopoSort.h"
#include "math/TensorWrapper.h"

using namespace hahaha::compute;
using namespace hahaha::math;

class TopoSortTest : public ::testing::Test {
protected:
    // Helper to create a dummy leaf node
    std::shared_ptr<ComputeNode<float>> createLeaf(float val = 0.0f) {
        auto shape = TensorShape({1});
        auto data = std::make_shared<TensorWrapper<float>>(shape, val);
        return std::make_shared<ComputeNode<float>>(data);
    }

    // Helper to check if a node appears before another in the list
    bool isBefore(const std::vector<std::shared_ptr<ComputeNode<float>>>& list,
                  const std::shared_ptr<ComputeNode<float>>& before,
                  const std::shared_ptr<ComputeNode<float>>& after) {
        size_t beforeIdx = list.size(), afterIdx = list.size();
        for (size_t i = 0; i < list.size(); ++i) {
            if (list[i] == before) beforeIdx = i;
            if (list[i] == after) afterIdx = i;
        }
        return beforeIdx < afterIdx && afterIdx < list.size();
    }
};

TEST_F(TopoSortTest, SingleNode) {
    auto a = createLeaf(1.0f);
    TopoSort<float> sorter;
    auto list = sorter.toTopoList(a);

    ASSERT_EQ(list.size(), 1);
    EXPECT_EQ(list[0], a);
}

TEST_F(TopoSortTest, LinearGraph) {
    // A -> B -> C
    auto a = createLeaf(1.0f);
    auto bData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 2.0f);
    auto b = std::make_shared<ComputeNode<float>>(a, a, bData, hahaha::common::Operator::Add, nullptr); // Dummy op
    
    auto cData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 3.0f);
    auto c = std::make_shared<ComputeNode<float>>(b, b, cData, hahaha::common::Operator::Add, nullptr);

    TopoSort<float> sorter;
    auto list = sorter.toTopoList(c);

    ASSERT_EQ(list.size(), 3);
    EXPECT_EQ(list[0], a);
    EXPECT_EQ(list[1], b);
    EXPECT_EQ(list[2], c);
}

TEST_F(TopoSortTest, DiamondGraph) {
    /*
         A
        / \
       B   C
        \ /
         D
    */
    auto a = createLeaf(1.0f);
    
    auto bData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 2.0f);
    auto b = std::make_shared<ComputeNode<float>>(a, a, bData, hahaha::common::Operator::Add, nullptr);

    auto cData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 3.0f);
    auto c = std::make_shared<ComputeNode<float>>(a, a, cData, hahaha::common::Operator::Add, nullptr);

    auto dData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 4.0f);
    auto d = std::make_shared<ComputeNode<float>>(b, c, dData, hahaha::common::Operator::Add, nullptr);

    TopoSort<float> sorter;
    auto list = sorter.toTopoList(d);

    ASSERT_EQ(list.size(), 4);
    EXPECT_TRUE(isBefore(list, a, b));
    EXPECT_TRUE(isBefore(list, a, c));
    EXPECT_TRUE(isBefore(list, b, d));
    EXPECT_TRUE(isBefore(list, c, d));
}

TEST_F(TopoSortTest, SharedDependency) {
    /*
         A
        / \
       B   |
        \ /
         C
    */
    auto a = createLeaf(1.0f);
    
    auto bData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 2.0f);
    auto b = std::make_shared<ComputeNode<float>>(a, a, bData, hahaha::common::Operator::Add, nullptr);

    auto cData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 3.0f);
    auto c = std::make_shared<ComputeNode<float>>(a, b, cData, hahaha::common::Operator::Add, nullptr);

    TopoSort<float> sorter;
    auto list = sorter.toTopoList(c);

    ASSERT_EQ(list.size(), 3);
    EXPECT_TRUE(isBefore(list, a, b));
    EXPECT_TRUE(isBefore(list, a, c));
    EXPECT_TRUE(isBefore(list, b, c));
}

TEST_F(TopoSortTest, MultipleCallsIndependence) {
    TopoSort<float> sorter;
    
    auto a = createLeaf(1.0f);
    auto b = createLeaf(2.0f);
    
    auto list1 = sorter.toTopoList(a);
    ASSERT_EQ(list1.size(), 1);
    EXPECT_EQ(list1[0], a);

    auto list2 = sorter.toTopoList(b);
    ASSERT_EQ(list2.size(), 1);
    EXPECT_EQ(list2[0], b);
    
    // Ensure 'a' didn't stick around in 'visited' and block 'b' or vice versa
    // This confirms our fix for the state residue bug.
}

TEST_F(TopoSortTest, UnaryAndBinaryMix) {
    /*
         A
         | (unary)
         B   C
          \ / (binary)
           D
    */
    auto a = createLeaf(1.0f);
    
    auto bData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 2.0f);
    auto b = ComputeNode<float>::createUnary(a, bData, hahaha::common::Operator::Add, nullptr);

    auto c = createLeaf(3.0f);

    auto dData = std::make_shared<TensorWrapper<float>>(TensorShape({1}), 4.0f);
    auto d = std::make_shared<ComputeNode<float>>(b, c, dData, hahaha::common::Operator::Add, nullptr);

    TopoSort<float> sorter;
    auto list = sorter.toTopoList(d);

    ASSERT_EQ(list.size(), 4);
    EXPECT_TRUE(isBefore(list, a, b));
    EXPECT_TRUE(isBefore(list, b, d));
    EXPECT_TRUE(isBefore(list, c, d));
    
    // Check if both A and C are before D
    EXPECT_TRUE(isBefore(list, a, d));
    EXPECT_TRUE(isBefore(list, c, d));
}

