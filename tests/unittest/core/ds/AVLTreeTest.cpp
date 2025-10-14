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
#include "core/ds/avlTree.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace hahaha::core::ds;

template <typename T>
class AVLTreeTest : public ::testing::Test {
protected:
    AVLTree<T> tree;

    // Helper to verify AVL tree height property
    int checkHeight(AVLTreeNode<T>* node) {
        if (!node) return 0;
        
        int leftHeight = checkHeight(node->getLeft());
        int rightHeight = checkHeight(node->getRight());
        
        if (leftHeight == -1 || rightHeight == -1)
            return -1;
        
        if (std::abs(leftHeight - rightHeight) > 1)
            return -1; // Balance violation
        
        return std::max(leftHeight, rightHeight) + 1;
    }

    // Verify node heights are correctly maintained
    bool verifyNodeHeights(AVLTreeNode<T>* node) {
        if (!node) return true;
        
        int leftHeight = node->getLeft() ? node->getLeft()->getHeight() : 0;
        int rightHeight = node->getRight() ? node->getRight()->getHeight() : 0;
        int expectedHeight = std::max(leftHeight, rightHeight) + 1;
        
        if (node->getHeight() != expectedHeight)
            return false;
        
        return verifyNodeHeights(node->getLeft()) && verifyNodeHeights(node->getRight());
    }

    // Verify BST property
    bool isBST(AVLTreeNode<T>* node, T min, T max) {
        if (!node) return true;
        if (node->getData() <= min || node->getData() >= max) return false;
        return isBST(node->getLeft(), min, node->getData()) && 
               isBST(node->getRight(), node->getData(), max);
    }

    // Verify balance factor
    bool verifyBalanceFactor(AVLTreeNode<T>* node) {
        if (!node) return true;
        
        int balanceFactor = node->getBalanceFactor();
        if (balanceFactor < -1 || balanceFactor > 1)
            return false;
        
        return verifyBalanceFactor(node->getLeft()) && verifyBalanceFactor(node->getRight());
    }

    void verifyProperties() {
        ASSERT_NE(checkHeight(tree.getRoot()), -1);
        ASSERT_TRUE(verifyNodeHeights(tree.getRoot()));
        ASSERT_TRUE(verifyBalanceFactor(tree.getRoot()));
        if (tree.getRoot()) {
            ASSERT_TRUE(isBST(tree.getRoot(), std::numeric_limits<T>::lowest(), 
                            std::numeric_limits<T>::max()));
        }
    }
};

using MyTypes = ::testing::Types<int, double>;
TYPED_TEST_SUITE(AVLTreeTest, MyTypes);

TYPED_TEST(AVLTreeTest, InitialState) {
    ASSERT_EQ(this->tree.getRoot(), nullptr);
    this->verifyProperties();
}

TYPED_TEST(AVLTreeTest, SimpleInsert) {
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(5);
    this->verifyProperties();
    ASSERT_TRUE(this->tree.find(10));
    ASSERT_TRUE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(5));
    ASSERT_FALSE(this->tree.find(15));
}

TYPED_TEST(AVLTreeTest, LeftLeftRotation) {
    // Insert in descending order to trigger LL rotation
    this->tree.insert(30);
    this->tree.insert(20);
    this->tree.insert(10);
    this->verifyProperties();
    
    // After rotation, 20 should be root
    ASSERT_EQ(this->tree.getRoot()->getData(), 20);
}

TYPED_TEST(AVLTreeTest, RightRightRotation) {
    // Insert in ascending order to trigger RR rotation
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);
    this->verifyProperties();
    
    // After rotation, 20 should be root
    ASSERT_EQ(this->tree.getRoot()->getData(), 20);
}

TYPED_TEST(AVLTreeTest, LeftRightRotation) {
    this->tree.insert(30);
    this->tree.insert(10);
    this->tree.insert(20);
    this->verifyProperties();
    
    // After LR rotation, 20 should be root
    ASSERT_EQ(this->tree.getRoot()->getData(), 20);
}

TYPED_TEST(AVLTreeTest, RightLeftRotation) {
    this->tree.insert(10);
    this->tree.insert(30);
    this->tree.insert(20);
    this->verifyProperties();
    
    // After RL rotation, 20 should be root
    ASSERT_EQ(this->tree.getRoot()->getData(), 20);
}

TYPED_TEST(AVLTreeTest, ComplexRotations) {
    std::vector<TypeParam> data = {10, 20, 30, 40, 50, 25};
    for (const auto& val : data) {
        this->tree.insert(val);
        this->verifyProperties();
    }
}

TYPED_TEST(AVLTreeTest, InsertDuplicate) {
    this->tree.insert(10);
    this->tree.insert(10);
    this->tree.insert(10);
    ASSERT_EQ(this->tree.find(10)->getData(), 10);
    this->verifyProperties();
}

TYPED_TEST(AVLTreeTest, SimpleRemove) {
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

TYPED_TEST(AVLTreeTest, RemoveNodeWithTwoChildren) {
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

TYPED_TEST(AVLTreeTest, RemoveRoot) {
    this->tree.insert(10);
    this->tree.insert(5);
    this->tree.insert(15);
    
    this->tree.remove(10);
    ASSERT_FALSE(this->tree.find(10));
    this->verifyProperties();
    ASSERT_NE(this->tree.getRoot(), nullptr);
}

TYPED_TEST(AVLTreeTest, RemoveTriggersRebalance) {
    // Build a tree that needs rebalancing after deletion
    for (int i = 1; i <= 7; i++) {
        this->tree.insert(i);
    }
    this->verifyProperties();
    
    // Remove nodes that cause rebalancing
    this->tree.remove(1);
    this->verifyProperties();
    
    this->tree.remove(2);
    this->verifyProperties();
}

TYPED_TEST(AVLTreeTest, StressTestInsertRemove) {
    std::vector<TypeParam> data(100);
    std::iota(data.begin(), data.end(), 1);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);

    for (const auto& val : data) {
        this->tree.insert(val);
        this->verifyProperties();
    }
    
    std::shuffle(data.begin(), data.end(), g);

    for (const auto& val : data) {
        this->tree.remove(val);
        this->verifyProperties();
    }
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

// Additional comprehensive tests

TYPED_TEST(AVLTreeTest, EmptyTreeOperations) {
    ASSERT_EQ(this->tree.getRoot(), nullptr);
    ASSERT_EQ(this->tree.find(10), nullptr);
    ASSERT_EQ(this->tree.min(), nullptr);
    ASSERT_EQ(this->tree.max(), nullptr);
    
    this->tree.remove(10); // Should not crash
    this->verifyProperties();
}

TYPED_TEST(AVLTreeTest, SingleNodeTree) {
    this->tree.insert(42);
    ASSERT_NE(this->tree.getRoot(), nullptr);
    ASSERT_EQ(this->tree.getRoot()->getData(), 42);
    ASSERT_EQ(this->tree.min()->getData(), 42);
    ASSERT_EQ(this->tree.max()->getData(), 42);
    
    this->tree.remove(42);
    ASSERT_EQ(this->tree.getRoot(), nullptr);
    this->verifyProperties();
}

TYPED_TEST(AVLTreeTest, MinMaxOperations) {
    std::vector<TypeParam> data = {50, 25, 75, 10, 30, 60, 80, 5, 15};
    for (const auto& val : data) {
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

TYPED_TEST(AVLTreeTest, SequentialInsertAscending) {
    for (int i = 1; i <= 20; i++) {
        this->tree.insert(i);
        this->verifyProperties();
    }
    
    ASSERT_EQ(this->tree.min()->getData(), 1);
    ASSERT_EQ(this->tree.max()->getData(), 20);
    
    for (int i = 1; i <= 20; i++) {
        ASSERT_TRUE(this->tree.find(i));
    }
}

TYPED_TEST(AVLTreeTest, SequentialInsertDescending) {
    for (int i = 20; i >= 1; i--) {
        this->tree.insert(i);
        this->verifyProperties();
    }
    
    ASSERT_EQ(this->tree.min()->getData(), 1);
    ASSERT_EQ(this->tree.max()->getData(), 20);
    
    for (int i = 1; i <= 20; i++) {
        ASSERT_TRUE(this->tree.find(i));
    }
}

TYPED_TEST(AVLTreeTest, FindNonExistent) {
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);
    
    ASSERT_FALSE(this->tree.find(5));
    ASSERT_FALSE(this->tree.find(15));
    ASSERT_FALSE(this->tree.find(25));
    ASSERT_FALSE(this->tree.find(35));
}

TYPED_TEST(AVLTreeTest, RemoveNonExistent) {
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

TYPED_TEST(AVLTreeTest, RemoveAllFromLeft) {
    for (int i = 1; i <= 10; i++) {
        this->tree.insert(i);
    }
    
    for (int i = 1; i <= 10; i++) {
        this->tree.remove(i);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(i));
    }
    
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(AVLTreeTest, RemoveAllFromRight) {
    for (int i = 1; i <= 10; i++) {
        this->tree.insert(i);
    }
    
    for (int i = 10; i >= 1; i--) {
        this->tree.remove(i);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(i));
    }
    
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(AVLTreeTest, RemoveAllFromMiddle) {
    for (int i = 1; i <= 10; i++) {
        this->tree.insert(i);
    }
    
    // Remove from middle outward
    std::vector<int> removeOrder = {5, 6, 4, 7, 3, 8, 2, 9, 1, 10};
    for (int val : removeOrder) {
        this->tree.remove(val);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(val));
    }
    
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(AVLTreeTest, InterleavedInsertRemove) {
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

TYPED_TEST(AVLTreeTest, LargeScaleRandom) {
    std::vector<TypeParam> data(500);
    std::iota(data.begin(), data.end(), 1);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    // Insert all
    for (const auto& val : data) {
        this->tree.insert(val);
    }
    this->verifyProperties();
    
    // Verify all present
    for (const auto& val : data) {
        ASSERT_TRUE(this->tree.find(val));
    }
    
    // Remove half randomly
    std::shuffle(data.begin(), data.end(), g);
    for (size_t i = 0; i < data.size() / 2; i++) {
        this->tree.remove(data[i]);
    }
    this->verifyProperties();
    
    // Verify removed ones are gone
    for (size_t i = 0; i < data.size() / 2; i++) {
        ASSERT_FALSE(this->tree.find(data[i]));
    }
    
    // Verify remaining ones are still there
    for (size_t i = data.size() / 2; i < data.size(); i++) {
        ASSERT_TRUE(this->tree.find(data[i]));
    }
}

TYPED_TEST(AVLTreeTest, RemoveRootRepeatedly) {
    for (int i = 1; i <= 10; i++) {
        this->tree.insert(i);
    }
    
    while (this->tree.getRoot() != nullptr) {
        auto rootData = this->tree.getRoot()->getData();
        this->tree.remove(rootData);
        this->verifyProperties();
        ASSERT_FALSE(this->tree.find(rootData));
    }
    
    ASSERT_EQ(this->tree.getRoot(), nullptr);
}

TYPED_TEST(AVLTreeTest, HeightConsistency) {
    for (int i = 1; i <= 50; i++) {
        this->tree.insert(i);
    }
    
    // Verify height is balanced (log n)
    int height = this->checkHeight(this->tree.getRoot());
    ASSERT_NE(height, -1);
    ASSERT_LE(height, 8); // For 50 nodes, height should be <= 1.44*log2(50) ≈ 8
    
    // Remove some and verify again
    for (int i = 1; i <= 25; i++) {
        this->tree.remove(i);
        int newHeight = this->checkHeight(this->tree.getRoot());
        ASSERT_NE(newHeight, -1);
    }
}

TYPED_TEST(AVLTreeTest, DuplicateInsertionStability) {
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);
    
    // Insert duplicates (should be no-ops)
    this->tree.insert(10);
    this->tree.insert(20);
    this->tree.insert(30);
    
    this->verifyProperties();
    
    ASSERT_TRUE(this->tree.find(10));
    ASSERT_TRUE(this->tree.find(20));
    ASSERT_TRUE(this->tree.find(30));
}

TYPED_TEST(AVLTreeTest, AlternatingInsertRemove) {
    for (int i = 1; i <= 20; i++) {
        this->tree.insert(i * 2);
        this->verifyProperties();
        
        if (i > 1) {
            this->tree.remove((i - 1) * 2);
            this->verifyProperties();
        }
    }
    
    // Should have only the last inserted element
    ASSERT_TRUE(this->tree.find(40));
    for (int i = 1; i < 20; i++) {
        ASSERT_FALSE(this->tree.find(i * 2));
    }
}

TYPED_TEST(AVLTreeTest, BalanceFactorAfterOperations) {
    // Insert many elements
    for (int i = 1; i <= 31; i++) {
        this->tree.insert(i);
    }
    
    // Every node should have balance factor in [-1, 1]
    ASSERT_TRUE(this->verifyBalanceFactor(this->tree.getRoot()));
    
    // Remove some
    for (int i = 1; i <= 15; i++) {
        this->tree.remove(i);
        ASSERT_TRUE(this->verifyBalanceFactor(this->tree.getRoot()));
    }
}

TYPED_TEST(AVLTreeTest, HeightPropertyMaintained) {
    for (int i = 1; i <= 20; i++) {
        this->tree.insert(i);
        ASSERT_TRUE(this->verifyNodeHeights(this->tree.getRoot()));
    }
    
    for (int i = 1; i <= 10; i++) {
        this->tree.remove(i);
        ASSERT_TRUE(this->verifyNodeHeights(this->tree.getRoot()));
    }
}

