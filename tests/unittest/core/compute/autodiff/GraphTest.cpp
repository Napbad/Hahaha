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
#include "include/ml/common/TensorVar.h"
#include "core/compute/autodiff/ops.h"
#include "core/ds/String.h"

using namespace hahaha;
using namespace hahaha::ad;
using hahaha::core::ds::String;

class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        a = std::make_shared<TensorVar<float>>(1.0f, String("a"));
        b = std::make_shared<TensorVar<float>>(2.0f, String("b"));
        d = std::make_shared<TensorVar<float>>(3.0f, String("d"));
        v1 = std::make_shared<TensorVar<float>>(1.0f, String("v1"));
        v2 = std::make_shared<TensorVar<float>>(2.0f, String("v2"));
        v3 = std::make_shared<TensorVar<float>>(3.0f, String("v3"));
        v4 = std::make_shared<TensorVar<float>>(4.0f, String("v4"));
    }

    std::shared_ptr<GraphNode<float>> a, b, d;
    std::shared_ptr<GraphNode<float>> v1, v2, v3, v4;
};


TEST_F(GraphTest, SimpleAddition) {
    auto c = a + b;

    // The result should be a ComputeNode, specifically an AddNode
    auto add_node = std::dynamic_pointer_cast<AddNode<float>>(c);
    ASSERT_NE(add_node, nullptr);

    // It should have two sources
    ASSERT_EQ(add_node->srcs().size(), 2);

    // The sources should be the original 'a' and 'b' nodes
    ASSERT_EQ(add_node->srcs()[0], a);
    ASSERT_EQ(add_node->srcs()[1], b);
}

TEST_F(GraphTest, SimpleMultiplication) {
    auto c = b * d;

    auto mul_node = std::dynamic_pointer_cast<MultiplyNode<float>>(c);
    ASSERT_NE(mul_node, nullptr);
    ASSERT_EQ(mul_node->srcs().size(), 2);
    ASSERT_EQ(mul_node->srcs()[0], b);
    ASSERT_EQ(mul_node->srcs()[1], d);
}

TEST_F(GraphTest, ChainedOperations) {
    // c = a + (b * d)
    auto c = a + b * d;

    // Top node should be an AddNode
    auto add_node = std::dynamic_pointer_cast<AddNode<float>>(c);
    ASSERT_NE(add_node, nullptr);
    ASSERT_EQ(add_node->srcs().size(), 2);

    // The first source of the addition should be 'a'
    ASSERT_EQ(add_node->srcs()[0], a);

    // The second source should be the result of the multiplication
    auto mul_node = std::dynamic_pointer_cast<MultiplyNode<float>>(add_node->srcs()[1]);
    ASSERT_NE(mul_node, nullptr);
    ASSERT_EQ(mul_node->srcs().size(), 2);

    // The sources of the multiplication should be 'b' and 'd'
    ASSERT_EQ(mul_node->srcs()[0], b);
    ASSERT_EQ(mul_node->srcs()[1], d);
}

TEST_F(GraphTest, ComplexGraphStructure) {
    // result = (v1 * v2) + (v3 * v4)
    auto result = v1 * v2 + v3 * v4;

    // Top node is an AddNode
    auto add_node = std::dynamic_pointer_cast<AddNode<float>>(result);
    ASSERT_NE(add_node, nullptr);
    ASSERT_EQ(add_node->srcs().size(), 2);

    // Left source is the first multiplication
    auto mul_node1 = std::dynamic_pointer_cast<MultiplyNode<float>>(add_node->srcs()[0]);
    ASSERT_NE(mul_node1, nullptr);
    ASSERT_EQ(mul_node1->srcs().size(), 2);
    ASSERT_EQ(mul_node1->srcs()[0], v1);
    ASSERT_EQ(mul_node1->srcs()[1], v2);

    // Right source is the second multiplication
    auto mul_node2 = std::dynamic_pointer_cast<MultiplyNode<float>>(add_node->srcs()[1]);
    ASSERT_NE(mul_node2, nullptr);
    ASSERT_EQ(mul_node2->srcs().size(), 2);
    ASSERT_EQ(mul_node2->srcs()[0], v3);
    ASSERT_EQ(mul_node2->srcs()[1], v4);
}
