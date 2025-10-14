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
// Created by Napbad on 8/26/25.
//

#ifndef HIAHIAHIA_RBTREE_H
#define HIAHIAHIA_RBTREE_H
#include <cassert>

namespace hahaha::core::ds
{

enum class Color
{
    Red,
    Black
};

template <typename T> class RBTreeNode
{
  public:
    explicit RBTreeNode(const T& data) : data_(data)
    {
    }

    RBTreeNode(const T& data, RBTreeNode* parent) : parent_(parent), data_(data)
    {
    }

    ~RBTreeNode()
    {
        delete left_;
        delete right_;
    }

    void setBlack()
    {
        color_ = Color::Black;
    }
    void setRed()
    {
        color_ = Color::Red;
    }
    void setColor(Color color)
    {
        color_ = color;
    }
    [[nodiscard]] Color getColor() const
    {
        return color_;
    }
    T getData() const
    {
        return data_;
    }
    T& getDataRef()
    {
        return data_;
    }
    const T& getDataConstRef() const
    {
        return data_;
    }
    void setData(const T& data)
    {
        data_ = data;
    }
    RBTreeNode* getLeft() const
    {
        return left_;
    }
    RBTreeNode* getRight() const
    {
        return right_;
    }
    RBTreeNode* getParent() const
    {
        return parent_;
    }

    RBTreeNode* getAnotherChild(RBTreeNode* node) const
    {
        if (this->left_ == node)
            return this->getRight();
        if (this->right_ == node)
            return this->getLeft();
        return this->getLeft(); // default
    }

    bool hasRight()
    {
        return right_ != nullptr;
    }
    bool hasLeft()
    {
        return left_ != nullptr;
    }

    bool hasChild()
    {
        return hasLeft() || hasRight();
    }

    void setLeft(RBTreeNode* node)
    {
        left_ = node;
        if (left_)
            left_->parent_ = this;
    }
    void setRight(RBTreeNode* node)
    {
        right_ = node;
        if (right_)
            right_->parent_ = this;
    }
    void setParent(RBTreeNode* node)
    {
        parent_ = node;
    }

    void setChild(RBTreeNode* node)
    {
        if (node->getData() < this->getData())
            this->setLeft(node);
        else if (node->getData() > this->getData())
            this->setRight(node);
    }

    void insertToNode(T val)
    {
        if (data_ < val)
        {
            if (right_)
                right_->insertToNode(val);
            else
                right_ = new RBTreeNode(val, this);
        }
        if (data_ > val)
        {
            if (left_)
                left_->insertToNode(val);
            else
                left_ = new RBTreeNode(val, this);
        }
    }

    bool isBlack() const
    {
        return color_ == Color::Black;
    }
    bool isRed() const
    {
        return color_ == Color::Red;
    }

  private:
    Color color_ = Color::Red; // default color of node that inserted is red
    RBTreeNode* left_ = nullptr;
    RBTreeNode* right_ = nullptr;
    RBTreeNode* parent_ = nullptr;

    T data_;
};
// Red-Black Tree
template <typename T> class RBTree
{
  public:
    class iterator
    {
    };

    RBTree() = default;
    ~RBTree()
    {
        delete root_;
    }

    void insert(T val)
    {
        // inserting:
        if (!root_)
            return insertToRoot(val);
        auto targetNode = insertToNode(root_, val);
        if (!targetNode)
            return;

        adjustTree(targetNode);
    }

    void insertToRoot(T val)
    {
        if (root_)
            delete root_;
        root_ = new RBTreeNode<T>(val);
        root_->setBlack();
    }

    // return the insert result of the target place to insert
    // nullptr will be returned if no position to insert
    // (already have the same val)
    RBTreeNode<T>* insertToNode(RBTreeNode<T>* node, T val)
    {
        assert(node != nullptr);
        // node will never be a nullptr
        auto parent = node;
        while (node)
        {
            parent = node;
            if (val < node->getData())
                node = node->getLeft();
            else if (val > node->getData())
                node = node->getRight();
            else
                return nullptr;
        }

        if (val < parent->getData())
        {
            parent->setLeft(new RBTreeNode<T>(val, parent));
            return parent->getLeft();
        }
        if (val > parent->getData())
        {
            parent->setRight(new RBTreeNode<T>(val, parent));
            return parent->getRight();
        }

        return nullptr;
    }

    // from this node, adjust the tree
    void adjustTree(RBTreeNode<T>* node)
    {
        // node will never be a nullptr, and the node need to adjust is always
        // be a red node
        assert(node != nullptr && node->isRed());
        if (node == root_)
        {
            root_->setBlack();
            return;
        }
        auto parent = node->getParent();
        if (parent->isBlack())
            return;
        // node will always have a grandparent(if it does not have,
        // then its father will be root which definitely a black node)
        auto grandparent = parent->getParent();
        // if it does not have a grandparent, then it is a child of root, and
        // the root is black this is red, no errors
        if (!grandparent)
            return;

        if (auto uncle = grandparent->getAnotherChild(parent); isRed(uncle))
            adjustWithRedUncle(node, uncle);
        else
            adjustWithBlackUncle(node);
    }

    void adjustWithRedUncle(RBTreeNode<T>* node, RBTreeNode<T>* uncle)
    {
        // in this case, the color of node, node's parend nad its uncle must be
        // red and node's parent's parent must be black
        node->getParent()->setBlack();
        node->getParent()->getParent()->setRed();
        uncle->setBlack();

        // now adjust the grandparent
        adjustTree(node->getParent()->getParent());
    }

    void adjustWithBlackUncle(RBTreeNode<T>* node)
    {
        if (isLL(node))
        {
            adjustWithBlackUncleAndLL(node);
        }
        else if (isLR(node))
        {
            adjustWithBlackUncleAndLR(node);
        }
        else if (isRL(node))
        {
            adjustWithBlackUncleAndRL(node);
        }
        else if (isRR(node))
        {
            adjustWithBlackUncleAndRR(node);
        }
    }
    void adjustWithBlackUncleAndLL(RBTreeNode<T>* node)
    {
        // current: node(the new inserted one): red, parent: red, uncle: black,
        // grandparent: black
        rightRotation(node->getParent()->getParent());
        node->getParent()->setBlack();           // previous parent node
        node->getParent()->getRight()->setRed(); // previous grandparent node
    }

    void adjustWithBlackUncleAndLR(RBTreeNode<T>* node)
    {
        // current: node(the new inserted one): red, parent: red, uncle: black,
        // grandparent: black
        leftRotation(node->getParent());
        // current node will at the position that its previous parent has
        rightRotation(node->getParent());
        // now this node is the parent of its previous parent(now left child)
        // and its previous grandparent (now right child)
        node->setBlack();
        node->getRight()->setRed();
    }

    void adjustWithBlackUncleAndRL(RBTreeNode<T>* node)
    {
        // current: node(the new inserted one): red, parent: red, uncle: black,
        // grandparent: black
        rightRotation(node->getParent());
        // current node will at the position that its previous parent has
        leftRotation(node->getParent());
        // now this node is the parent of its previous parent
        // and its previous grandparent
        node->setBlack();
        node->getLeft()->setRed();
    }

    void adjustWithBlackUncleAndRR(RBTreeNode<T>* node)
    {
        // current: node(the new inserted one): red, parent: red, uncle: black,
        // grandparent: black
        leftRotation(node->getParent()->getParent());
        node->getParent()->setBlack(); // previous parent node
        node->getParent()
            ->getAnotherChild(node)
            ->setRed(); // previous grandparent node
    }

    void remove(T val)
    {
        auto target = find(val);
        if (!target)
            return;

        // If target has two children, swap with successor
        if (target->getLeft() && target->getRight())
        {
            auto successor = findMinNode(target->getRight());
            swapNodeData(target, successor);
            target = successor;
        }

        // Now target has at most one child
        auto child = target->getLeft() ? target->getLeft() : target->getRight();

        if (target == root_)
        {
            setRoot(child);
            if (child)
                child->setBlack();
            target->setLeft(nullptr);
            target->setRight(nullptr);
            delete target;
            return;
        }

        if (target->isRed())
        {
            removeRedNode(target);
        }
        else if (child && child->isRed())
        {
            removeBlackNodeWithRedChild(target, child);
        }
        else
        {
            removeBlackNodeWithBlackChild(target, child);
        }
    }

    void removeRedNode(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        if (parent->getLeft() == node)
            parent->setLeft(nullptr);
        else
            parent->setRight(nullptr);

        node->setLeft(nullptr);
        node->setRight(nullptr);
        delete node;
    }

    void removeBlackNodeWithRedChild(RBTreeNode<T>* node, RBTreeNode<T>* child)
    {
        replaceNode(node, child);
        child->setBlack();
        node->setLeft(nullptr);
        node->setRight(nullptr);
        delete node;
    }

    void removeBlackNodeWithBlackChild(RBTreeNode<T>* node,
                                       RBTreeNode<T>* child)
    {
        auto parent = node->getParent();
        bool wasLeftChild = parent && isLeftChild(parent, node);
        replaceNode(node, child);

        if (child)
        {
            fixDoubleBlack(child);
        }
        else if (parent)
        {
            fixDoubleBlackNull(parent, wasLeftChild);
        }

        node->setLeft(nullptr);
        node->setRight(nullptr);
        delete node;
    }

    void fixDoubleBlack(RBTreeNode<T>* node)
    {
        if (node == root_)
            return;

        auto parent = node->getParent();
        auto sibling = getSibling(node);

        if (!sibling)
        {
            fixDoubleBlack(parent);
            return;
        }

        if (sibling->isRed())
        {
            handleRedSibling(node, parent, sibling);
        }
        else
        {
            handleBlackSibling(node, parent, sibling);
        }
    }

    void fixDoubleBlackNull(RBTreeNode<T>* parent, bool isLeft)
    {
        if (!parent)
            return;

        auto sibling = isLeft ? parent->getRight() : parent->getLeft();

        if (!sibling)
        {
            fixDoubleBlack(parent);
            return;
        }

        if (sibling->isRed())
        {
            handleRedSiblingNull(parent, sibling, isLeft);
        }
        else
        {
            handleBlackSiblingNull(parent, sibling, isLeft);
        }
    }

    void handleRedSibling(RBTreeNode<T>* node,
                          RBTreeNode<T>* parent,
                          RBTreeNode<T>* sibling)
    {
        parent->setRed();
        sibling->setBlack();

        if (isLeftChild(parent, node))
            leftRotation(parent);
        else
            rightRotation(parent);

        fixDoubleBlack(node);
    }

    void handleRedSiblingNull(RBTreeNode<T>* parent,
                              RBTreeNode<T>* sibling,
                              bool nodeIsLeft)
    {
        parent->setRed();
        sibling->setBlack();

        if (nodeIsLeft)
            leftRotation(parent);
        else
            rightRotation(parent);

        fixDoubleBlackNull(parent, nodeIsLeft);
    }

    void handleBlackSibling(RBTreeNode<T>* node,
                            RBTreeNode<T>* parent,
                            RBTreeNode<T>* sibling)
    {
        if (hasRedChild(sibling))
        {
            handleBlackSiblingWithRedChild(node, parent, sibling);
        }
        else
        {
            sibling->setRed();
            if (parent->isBlack())
                fixDoubleBlack(parent);
            else
                parent->setBlack();
        }
    }

    void handleBlackSiblingNull(RBTreeNode<T>* parent,
                                RBTreeNode<T>* sibling,
                                bool nodeIsLeft)
    {
        if (hasRedChild(sibling))
        {
            handleBlackSiblingWithRedChildNull(parent, sibling, nodeIsLeft);
        }
        else
        {
            sibling->setRed();
            if (parent->isBlack())
            {
                auto grandparent = parent->getParent();
                if (grandparent)
                    fixDoubleBlackNull(grandparent,
                                       isLeftChild(grandparent, parent));
            }
            else
            {
                parent->setBlack();
            }
        }
    }

    void handleBlackSiblingWithRedChild(RBTreeNode<T>* node,
                                        RBTreeNode<T>* parent,
                                        RBTreeNode<T>* sibling)
    {
        bool nodeIsLeft = isLeftChild(parent, node);

        if (nodeIsLeft)
        {
            if (sibling->getRight() && sibling->getRight()->isRed())
            {
                // RR case
                sibling->getRight()->setBlack();
                sibling->setColor(parent->getColor());
                parent->setBlack();
                leftRotation(parent);
            }
            else
            {
                // RL case
                sibling->getLeft()->setColor(parent->getColor());
                parent->setBlack();
                rightRotation(sibling);
                leftRotation(parent);
            }
        }
        else
        {
            if (sibling->getLeft() && sibling->getLeft()->isRed())
            {
                // LL case
                sibling->getLeft()->setBlack();
                sibling->setColor(parent->getColor());
                parent->setBlack();
                rightRotation(parent);
            }
            else
            {
                // LR case
                sibling->getRight()->setColor(parent->getColor());
                parent->setBlack();
                leftRotation(sibling);
                rightRotation(parent);
            }
        }
    }

    void handleBlackSiblingWithRedChildNull(RBTreeNode<T>* parent,
                                            RBTreeNode<T>* sibling,
                                            bool nodeIsLeft)
    {
        if (nodeIsLeft)
        {
            if (sibling->getRight() && sibling->getRight()->isRed())
            {
                // RR case
                sibling->getRight()->setBlack();
                sibling->setColor(parent->getColor());
                parent->setBlack();
                leftRotation(parent);
            }
            else
            {
                // RL case
                sibling->getLeft()->setColor(parent->getColor());
                parent->setBlack();
                rightRotation(sibling);
                leftRotation(parent);
            }
        }
        else
        {
            if (sibling->getLeft() && sibling->getLeft()->isRed())
            {
                // LL case
                sibling->getLeft()->setBlack();
                sibling->setColor(parent->getColor());
                parent->setBlack();
                rightRotation(parent);
            }
            else
            {
                // LR case
                sibling->getRight()->setColor(parent->getColor());
                parent->setBlack();
                leftRotation(sibling);
                rightRotation(parent);
            }
        }
    }

    RBTreeNode<T>* find(T val)
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

    const RBTreeNode<T>* find(T val) const
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

    RBTreeNode<T>* min()
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasLeft())
            cur = cur->getLeft();

        return cur;
    }

    const RBTreeNode<T>* min() const
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasLeft())
            cur = cur->getLeft();

        return cur;
    }

    RBTreeNode<T>* max()
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasRight())
            cur = cur->getRight();

        return cur;
    }

    const RBTreeNode<T>* max() const
    {
        if (!root_)
            return nullptr;

        auto cur = root_;
        while (cur->hasRight())
            cur = cur->getRight();

        return cur;
    }

    RBTreeNode<T>* getRoot() const
    {
        return root_;
    }

    // rotations are based on the node itself
    // that means this node have a ll or other unbalanced structure,
    void leftRotation(RBTreeNode<T>* node)
    {
        if (!node->getRight())
            return;

        // nodes need to change
        auto prevRight = node->getRight();
        auto prevLeft = node->getLeft();
        auto prevParent = node->getParent();
        auto leftOfPrevRight = prevRight->getLeft();

        prevRight->setLeft(node);
        node->setRight(leftOfPrevRight);

        if (!prevParent)
            setRoot(prevRight);
        else
            prevParent->setChild(prevRight);
    }

    void rightRotation(RBTreeNode<T>* node)
    {
        if (!node->getLeft())
            return;

        // nodes need to change
        auto prevRight = node->getRight();
        auto prevLeft = node->getLeft();
        auto prevParent = node->getParent();
        auto rightOfPrevLeft = prevLeft->getRight();

        prevLeft->setRight(node);
        node->setLeft(rightOfPrevLeft);

        if (!prevParent)
            setRoot(prevLeft);
        else
            prevParent->setChild(prevLeft);
    }

  private:
    RBTreeNode<T>* root_ = nullptr;

    void setRoot(RBTreeNode<T>* node)
    {
        root_ = node;
        if (root_)
            root_->setParent(nullptr);
    }

    RBTreeNode<T>* findMinNode(RBTreeNode<T>* node)
    {
        while (node && node->hasLeft())
            node = node->getLeft();
        return node;
    }

    void swapNodeData(RBTreeNode<T>* a, RBTreeNode<T>* b)
    {
        T temp = a->getData();
        a->setData(b->getData());
        b->setData(temp);
    }

    void replaceNode(RBTreeNode<T>* oldNode, RBTreeNode<T>* newNode)
    {
        auto parent = oldNode->getParent();
        if (!parent)
        {
            setRoot(newNode);
        }
        else if (parent->getLeft() == oldNode)
        {
            parent->setLeft(newNode);
        }
        else
        {
            parent->setRight(newNode);
        }
    }

    RBTreeNode<T>* getSibling(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        if (!parent)
            return nullptr;
        return parent->getAnotherChild(node);
    }

    bool hasRedChild(RBTreeNode<T>* node)
    {
        if (!node)
            return false;
        return (node->getLeft() && node->getLeft()->isRed())
            || (node->getRight() && node->getRight()->isRed());
    }

    bool isLeftChild(RBTreeNode<T>* parent, RBTreeNode<T>* child)
    {
        return parent && parent->getLeft() == child;
    }

    static bool isBlack(RBTreeNode<T>* node)
    {
        if (!node)
            return true;
        return node->isBlack();
    }

    static bool isRed(RBTreeNode<T>* node)
    {
        if (!node)
            return false;
        return node->isRed();
    }

    // Case: Parent is LEFT child of Grandparent, Node is LEFT child of Parent
    static bool isLL(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        auto grandparent = parent->getParent();
        return parent == grandparent->getLeft() && node == parent->getLeft();
    }

    // Case: Parent is LEFT child of Grandparent, Node is RIGHT child of Parent
    static bool isLR(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        auto grandparent = parent->getParent();
        return parent == grandparent->getLeft() && node == parent->getRight();
    }

    // Case: Parent is RIGHT child of Grandparent, Node is RIGHT child of Parent
    static bool isRR(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        auto grandparent = parent->getParent();
        return parent == grandparent->getRight() && node == parent->getRight();
    }

    // Case: Parent is RIGHT child of Grandparent, Node is LEFT child of Parent
    static bool isRL(RBTreeNode<T>* node)
    {
        auto parent = node->getParent();
        auto grandparent = parent->getParent();
        return parent == grandparent->getRight() && node == parent->getLeft();
    }
};
} // namespace hahaha::core::ds

#endif // HIAHIAHIA_RBTREE_H
