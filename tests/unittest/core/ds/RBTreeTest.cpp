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

#include "core/ds/rbTree.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace hahaha::core::ds;

template <typename T> class RBTreeTest : public ::testing::Test
{
  protected:
    RBTree<T> tree;

    // Helper to verify the Red-Black Tree properties
    // Property 1 (nodes are red or black) is guaranteed by the enum.
    // Property 2 (root is black)
    bool rootIsBlack()
    {
        if (!tree.getRoot())
            return true; // An empty tree is valid
        return tree.getRoot()->isBlack();
    }

    // Property 3 (all leaves are black) is implicit as nullptr is considered
    // black.

    // Property 4 (no two adjacent red nodes)
    bool noRedRedViolation(RBTreeNode<T>* node)
    {
        if (!node)
            return true;
        if (node->isRed())
        {
            if (isRed(node->getLeft()) || isRed(node->getRight()))
            {
                return false;
            }
        }
        return noRedRedViolation(node->getLeft())
            && noRedRedViolation(node->getRight());
    }

    // Property 5 (same black height on all paths)
    int blackHeight(RBTreeNode<T>* node)
    {
        if (!node)
            return 1; // Leaf nodes contribute 1 to black height
        int leftBlackHeight = blackHeight(node->getLeft());
        int rightBlackHeight = blackHeight(node->getRight());
        if (leftBlackHeight != rightBlackHeight)
        {
            return -1; // Indicates a violation
        }
        return (node->isBlack() ? 1 : 0) + leftBlackHeight;
    }

    bool isBST(RBTreeNode<T>* node, T min, T max)
    {
        if (!node)
            return true;
        if (node->getData() <= min || node->getData() >= max)
            return false;
        return isBST(node->getLeft(), min, node->getData())
            && isBST(node->getRight(), node->getData(), max);
    }

    static bool isRed(RBTreeNode<T>* node)
    {
        if (!node)
            return false;
        return node->isRed();
    }

    void verifyProperties()
    {
        ASSERT_TRUE(rootIsBlack());
        ASSERT_TRUE(noRedRedViolation(tree.getRoot()));
        ASSERT_NE(blackHeight(tree.getRoot()), -1);
        ASSERT_TRUE(isBST(tree.getRoot(),
                          std::numeric_limits<T>::min(),
                          std::numeric_limits<T>::max()));
    }
};

using MyTypes = ::testing::Types<int, double>;
TYPED_TEST_SUITE(RBTreeTest, MyTypes);

TYPED_TEST(RBTreeTest, InitialState)
{
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, SimpleInsert)
{
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(5);
    this->verifyProperties();
    ASSERT_TRUE(this->tree.find(10));
    ASSERT_TRUE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(5));
    ASSERT_FALSE(this->tree.find(15));
}

TYPED_TEST(RBTreeTest, InsertCausesRedUncleRecolor)
{
    // This sequence (10, 85, 15, 70, 20, 60, 30, 50) causes a red uncle case
    this->tree.insert(10);
    this->tree.insert(85);
    this->tree.insert(15);
    this->tree.insert(70);
    this->tree.insert(20);
    this->tree.insert(60);
    this->tree.insert(30);
    this->tree.insert(50);
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, InsertCausesRotations)
{
    // LL Case
    this->tree.insert(30);
    this->tree.insert(20);
    this->tree.insert(10);
    this->verifyProperties();

    // RR Case
    this->tree.insert(40);
    this->tree.insert(50);
    this->verifyProperties();

    // LR Case
    this->tree.insert(5);
    this->tree.insert(8);
    this->verifyProperties();

    // RL Case
    this->tree.insert(60);
    this->tree.insert(55);
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, InsertDuplicate)
{
    this->tree.insert(10);
    this->tree.insert(10);
    this->tree.insert(10);
    ASSERT_EQ(this->tree.find(10)->getData(), 10);
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, SimpleRemove)
{
    this->tree.insert(10);
    this->tree.insert(5);
    this->tree.insert(15);
    this->verifyProperties();

    this->tree.remove(5);
    ASSERT_FALSE(this->tree.find(5));
    this->verifyProperties();

    this->tree.remove(15);
    ASSERT_FALSE(this->tree.find(15));
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, RemoveNodeWithTwoChildren)
{
    this->tree.insert(10);
    this->tree.insert(5);
    this->tree.insert(20);
    this->tree.insert(15);
    this->tree.insert(25);
    this->verifyProperties();

    this->tree.remove(20);
    ASSERT_FALSE(this->tree.find(20));
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, RemoveRoot)
{
    this->tree.insert(10);
    this->tree.insert(5);
    this->tree.insert(15);

    this->tree.remove(10);
    ASSERT_FALSE(this->tree.find(10));
    this->verifyProperties();
    ASSERT_EQ(this->tree.getRoot()->getData(), 15);
}

TYPED_TEST(RBTreeTest, StressTestInsertRemove)
{
    std::vector<TypeParam> data(100);
    std::iota(data.begin(), data.end(), 1);
    std::random_shuffle(data.begin(), data.end());

    for (const auto& val : data)
    {
        this->tree.insert(val);
        this->verifyProperties();
    }

    std::random_shuffle(data.begin(), data.end());

    for (const auto& val : data)
    {
        this->tree.remove(val);
        this->verifyProperties();
    }
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

// Additional comprehensive tests

TYPED_TEST(RBTreeTest, EmptyTreeOperations)
{
    ASSERT_EQ(this->tree.getRoot(), nullptr);
    ASSERT_EQ(this->tree.find(10), nullptr);
    ASSERT_EQ(this->tree.min(), nullptr);
    ASSERT_EQ(this->tree.max(), nullptr);

    this->tree.remove(10); // Should not crash
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, SingleNodeTree)
{
    this->tree.insert(42);
    ASSERT_NE(this->tree.getRoot(), nullptr);
    ASSERT_TRUE(this->tree.getRoot()->isBlack());
    ASSERT_EQ(this->tree.getRoot()->getData(), 42);
    ASSERT_EQ(this->tree.min()->getData(), 42);
    ASSERT_EQ(this->tree.max()->getData(), 42);

    this->tree.remove(42);
    ASSERT_EQ(this->tree.getRoot(), nullptr);
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, MinMaxOperations)
{
    std::vector<TypeParam> data = {50, 25, 75, 10, 30, 60, 80, 5, 15};
    for (const auto& val : data)
    {
        this->tree.insert(val);
    }

    ASSERT_EQ(this->tree.min()->getData(), 5);
    ASSERT_EQ(this->tree.max()->getData(), 80);

    this->tree.remove(5);
    ASSERT_EQ(this->tree.min()->getData(), 10);

    this->tree.remove(80);
    ASSERT_EQ(this->tree.max()->getData(), 75);

    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, SequentialInsertAscending)
{
    for (int i = 1; i <= 20; i++)
    {
        this->tree.insert(i);
        this->verifyProperties();
    }

    ASSERT_EQ(this->tree.min()->getData(), 1);
    ASSERT_EQ(this->tree.max()->getData(), 20);

    for (int i = 1; i <= 20; i++)
    {
        ASSERT_TRUE(this->tree.find(i));
    }
}

TYPED_TEST(RBTreeTest, SequentialInsertDescending)
{
    for (int i = 20; i >= 1; i--)
    {
        this->tree.insert(i);
        this->verifyProperties();
    }

    ASSERT_EQ(this->tree.min()->getData(), 1);
    ASSERT_EQ(this->tree.max()->getData(), 20);

    for (int i = 1; i <= 20; i++)
    {
        ASSERT_TRUE(this->tree.find(i));
    }
}

TYPED_TEST(RBTreeTest, FindNonExistent)
{
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);

    ASSERT_FALSE(this->tree.find(5));
    ASSERT_FALSE(this->tree.find(15));
    ASSERT_FALSE(this->tree.find(25));
    ASSERT_FALSE(this->tree.find(35));
}

TYPED_TEST(RBTreeTest, RemoveNonExistent)
{
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);

    this->tree.remove(5);   // Should not crash
    this->tree.remove(15);  // Should not crash
    this->tree.remove(100); // Should not crash

    ASSERT_TRUE(this->tree.find(10));
    ASSERT_TRUE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(30));
    this->verifyProperties();
}

TYPED_TEST(RBTreeTest, RemoveAllFromLeft)
{
    for (int i = 1; i <= 10; i++)
    {
        this->tree.insert(i);
    }

    for (int i = 1; i <= 10; i++)
    {
        this->tree.remove(i);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(i));
    }

    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(RBTreeTest, RemoveAllFromRight)
{
    for (int i = 1; i <= 10; i++)
    {
        this->tree.insert(i);
    }

    for (int i = 10; i >= 1; i--)
    {
        this->tree.remove(i);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(i));
    }

    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(RBTreeTest, RemoveAllFromMiddle)
{
    for (int i = 1; i <= 10; i++)
    {
        this->tree.insert(i);
    }

    // Remove from middle outward
    std::vector<int> removeOrder = {5, 6, 4, 7, 3, 8, 2, 9, 1, 10};
    for (int val : removeOrder)
    {
        this->tree.remove(val);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(val));
    }

    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(RBTreeTest, InterleavedInsertRemove)
{
    // Insert, remove, insert pattern
    this->tree.insert(10);
    this->verifyProperties();

    this->tree.insert(20);
    this->verifyProperties();

    this->tree.remove(10);
    this->verifyProperties();

    this->tree.insert(30);
    this->verifyProperties();

    this->tree.insert(5);
    this->verifyProperties();

    this->tree.remove(20);
    this->verifyProperties();

    ASSERT_FALSE(this->tree.find(10));
    ASSERT_FALSE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(30));
    ASSERT_TRUE(this->tree.find(5));
}

TYPED_TEST(RBTreeTest, LargeScaleRandom)
{
    std::vector<TypeParam> data(500);
    std::iota(data.begin(), data.end(), 1);
    std::random_shuffle(data.begin(), data.end());

    // Insert all
    for (const auto& val : data)
    {
        this->tree.insert(val);
    }
    this->verifyProperties();

    // Verify all present
    for (const auto& val : data)
    {
        ASSERT_TRUE(this->tree.find(val));
    }

    // Remove half randomly
    std::random_shuffle(data.begin(), data.end());
    for (size_t i = 0; i < data.size() / 2; i++)
    {
        this->tree.remove(data[i]);
    }
    this->verifyProperties();

    // Verify removed ones are gone
    for (size_t i = 0; i < data.size() / 2; i++)
    {
        ASSERT_FALSE(this->tree.find(data[i]));
    }

    // Verify remaining ones are still there
    for (size_t i = data.size() / 2; i < data.size(); i++)
    {
        ASSERT_TRUE(this->tree.find(data[i]));
    }
}

TYPED_TEST(RBTreeTest, RemoveRootRepeatedly)
{
    // Build a tree
    for (int i = 1; i <= 10; i++)
    {
        this->tree.insert(i);
    }

    // Remove root repeatedly
    while (this->tree.getRoot() != nullptr)
    {
        auto rootData = this->tree.getRoot()->getData();
        this->tree.remove(rootData);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(rootData));
    }

    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(RBTreeTest, BlackHeightConsistency)
{
    // Insert many elements
    for (int i = 1; i <= 50; i++)
    {
        this->tree.insert(i);
    }

    // Verify black height is consistent
    int bh = this->blackHeight(this->tree.getRoot());
    ASSERT_NE(bh, -1);

    // Remove some and verify again
    for (int i = 1; i <= 25; i++)
    {
        this->tree.remove(i);
        int newBh = this->blackHeight(this->tree.getRoot());
        ASSERT_NE(newBh, -1);
    }
}

TYPED_TEST(RBTreeTest, RootColorAfterOperations)
{
    // Root should always be black
    this->tree.insert(10);
    ASSERT_TRUE(this->tree.getRoot()->isBlack());

    this->tree.insert(20);
    ASSERT_TRUE(this->tree.getRoot()->isBlack());

    this->tree.insert(30);
    ASSERT_TRUE(this->tree.getRoot()->isBlack());

    this->tree.remove(10);
    if (this->tree.getRoot())
        ASSERT_TRUE(this->tree.getRoot()->isBlack());

    this->tree.remove(20);
    if (this->tree.getRoot())
        ASSERT_TRUE(this->tree.getRoot()->isBlack());

    this->tree.remove(30);
    // Tree is now empty, no root to check
}

TYPED_TEST(RBTreeTest, DuplicateInsertionStability)
{
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);

    // Insert duplicates (should be no-ops)
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);

    this->verifyProperties();

    // Tree structure should remain valid
    ASSERT_TRUE(this->tree.find(10));
    ASSERT_TRUE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(30));
}

TYPED_TEST(RBTreeTest, AlternatingInsertRemove)
{
    for (int i = 1; i <= 20; i++)
    {
        this->tree.insert(i * 2);
        this->verifyProperties();

        if (i > 1)
        {
            this->tree.remove((i - 1) * 2);
            this->verifyProperties();
        }
    }

    // Should have only the last inserted element
    ASSERT_TRUE(this->tree.find(40));
    for (int i = 1; i < 20; i++)
    {
        ASSERT_FALSE(this->tree.find(i * 2));
    }
}
