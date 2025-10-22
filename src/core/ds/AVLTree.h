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

//
// Created by napbad on 10/14/25.
//

#ifndef HAHAHA_AVLTREE_H
#define HAHAHA_AVLTREE_H

#include <algorithm>
#include <cassert>

namespace hahaha::core::ds
{

template <typename T> class AVLTreeNode
{
  public:
    explicit AVLTreeNode(const T& data) : data_(data)
    {
    }

    AVLTreeNode(const T& data, AVLTreeNode* parent)
        : parent_(parent), data_(data)
    {
    }

    ~AVLTreeNode()
    {
        delete left_;
        delete right_;
    }

    T getData() const
    {
        return data_;
    }

    void setData(const T& data)
    {
        data_ = data;
    }

    AVLTreeNode* getLeft() const
    {
        return left_;
    }

    AVLTreeNode* getRight() const
    {
        return right_;
    }

    AVLTreeNode* getParent() const
    {
        return parent_;
    }

    [[nodiscard]] int getHeight() const
    {
        return height_;
    }

    void setLeft(AVLTreeNode* node)
    {
        left_ = node;
        if (left_)
            left_->parent_ = this;
    }

    void setRight(AVLTreeNode* node)
    {
        right_ = node;
        if (right_)
            right_->parent_ = this;
    }

    void setParent(AVLTreeNode* node)
    {
        parent_ = node;
    }

    void updateHeight()
    {
        const int leftHeight = left_ ? left_->height_ : 0;
        const int rightHeight = right_ ? right_->height_ : 0;
        height_ = std::max(leftHeight, rightHeight) + 1;
    }

    [[nodiscard]] int getBalanceFactor() const
    {
        const int leftHeight = left_ ? left_->height_ : 0;
        const int rightHeight = right_ ? right_->height_ : 0;
        return leftHeight - rightHeight;
    }

    [[nodiscard]] bool hasLeft() const
    {
        return left_ != nullptr;
    }

    [[nodiscard]] bool hasRight() const
    {
        return right_ != nullptr;
    }

    [[nodiscard]] bool hasChild() const
    {
        return hasLeft() || hasRight();
    }

  private:
    AVLTreeNode* left_ = nullptr;
    AVLTreeNode* right_ = nullptr;
    AVLTreeNode* parent_ = nullptr;
    T data_;
    int height_ = 1;
};

template <typename T> class AVLTree
{
  public:
    class iterator
    {
    };

    AVLTree() = default;
    ~AVLTree()
    {
        delete root_;
    }

    void insert(T val)
    {
        if (!root_)
        {
            root_ = new AVLTreeNode<T>(val);
            return;
        }

        auto newNode = insertToNode(root_, val);
        if (!newNode)
            return; // Duplicate value

        // Rebalance from the new node's parent upward
        auto current = newNode->getParent();
        while (current)
        {
            current->updateHeight();
            current = rebalance(current);
            current = current->getParent();
        }
    }

    void remove(T val)
    {
        auto target = find(val);
        if (!target)
            return;

        removeNode(target);
    }

    AVLTreeNode<T>* find(T val) const
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur != nullptr)
        {
            if (cur->getData() == val)
                return cur;
            if (cur->getData() > val)
                cur = cur->getLeft();
            else
                cur = cur->getRight();
        }
        return nullptr;
    }

    AVLTreeNode<T>* min() const
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasLeft())
            cur = cur->getLeft();

        return cur;
    }

    AVLTreeNode<T>* max() const
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasRight())
            cur = cur->getRight();

        return cur;
    }

    AVLTreeNode<T>* getRoot() const
    {
        return root_;
    }

  private:
    AVLTreeNode<T>* root_ = nullptr;

    AVLTreeNode<T>* insertToNode(AVLTreeNode<T>* node, T val)
    {
        assert(node != nullptr);

        auto parent = node;
        while (node)
        {
            parent = node;
            if (val < node->getData())
                node = node->getLeft();
            else if (val > node->getData())
                node = node->getRight();
            else
                return nullptr; // Duplicate
        }

        if (val < parent->getData())
        {
            parent->setLeft(new AVLTreeNode<T>(val, parent));
            return parent->getLeft();
        }
        parent->setRight(new AVLTreeNode<T>(val, parent));
        return parent->getRight();
    }

    void removeNode(AVLTreeNode<T>* node)
    {
        // Case 1: Node with two children - replace with successor
        if (node->getLeft() && node->getRight())
        {
            auto successor = findMinNode(node->getRight());
            node->setData(successor->getData());
            removeNode(successor);
            return;
        }

        // Case 2: Node with at most one child
        auto child = node->getLeft() ? node->getLeft() : node->getRight();
        auto parent = node->getParent();

        if (node == root_)
        {
            root_ = child;
            if (root_)
                root_->setParent(nullptr);
        }
        else
        {
            if (parent->getLeft() == node)
                parent->setLeft(child);
            else
                parent->setRight(child);
        }

        // Clean up and rebalance from parent upward
        node->setLeft(nullptr);
        node->setRight(nullptr);
        delete node;

        // Rebalance from parent upward
        auto current = parent;
        while (current)
        {
            current->updateHeight();
            current = rebalance(current);
            current = current->getParent();
        }
    }

    AVLTreeNode<T>* rebalance(AVLTreeNode<T>* node)
    {

        // Left heavy
        if (const int balanceFactor = node->getBalanceFactor(); balanceFactor > 1)
        {
            // Left-Right case
            if (node->getLeft() && node->getLeft()->getBalanceFactor() < 0)
            {
                leftRotate(node->getLeft());
            }
            // Left-Left case
            return rightRotate(node);
        }
        // Right heavy
        else if (balanceFactor < -1)
        {
            // Right-Left case
            if (node->getRight() && node->getRight()->getBalanceFactor() > 0)
            {
                rightRotate(node->getRight());
            }
            // Right-Right case
            return leftRotate(node);
        }

        return node;
    }

    AVLTreeNode<T>* leftRotate(AVLTreeNode<T>* node)
    {
        auto rightChild = node->getRight();
        if (!rightChild)
            return node;

        auto parent = node->getParent();
        auto leftOfRight = rightChild->getLeft();

        // Perform rotation
        rightChild->setLeft(node);
        node->setRight(leftOfRight);

        // Update heights
        node->updateHeight();
        rightChild->updateHeight();

        // Update parent connection
        if (!parent)
        {
            root_ = rightChild;
            root_->setParent(nullptr);
        }
        else
        {
            if (parent->getLeft() == node)
                parent->setLeft(rightChild);
            else
                parent->setRight(rightChild);
        }

        return rightChild;
    }

    AVLTreeNode<T>* rightRotate(AVLTreeNode<T>* node)
    {
        auto leftChild = node->getLeft();
        if (!leftChild)
            return node;

        auto parent = node->getParent();
        auto rightOfLeft = leftChild->getRight();

        // Perform rotation
        leftChild->setRight(node);
        node->setLeft(rightOfLeft);

        // Update heights
        node->updateHeight();
        leftChild->updateHeight();

        // Update parent connection
        if (!parent)
        {
            root_ = leftChild;
            root_->setParent(nullptr);
        }
        else
        {
            if (parent->getLeft() == node)
                parent->setLeft(leftChild);
            else
                parent->setRight(leftChild);
        }

        return leftChild;
    }

    static AVLTreeNode<T>* findMinNode(AVLTreeNode<T>* node)
    {
        while (node && node->hasLeft())
            node = node->getLeft();
        return node;
    }
};

} // namespace hahaha::core::ds

#endif // HAHAHA_AVLTREE_H
