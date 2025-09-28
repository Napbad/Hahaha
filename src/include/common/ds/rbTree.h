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

namespace hahaha::common::ds {

  enum class Color { Red, Black };

  template<typename T>
  class RBTreeNode {
public:
    explicit RBTreeNode(const T &data) : data(data) {}
    ~RBTreeNode() {
      delete left;
      delete right;
    }

    void setBlack() { color = Color::Black; }
    void setRed() { color = Color::Red; }
    [[nodiscard]] Color getColor() const { return color; }
    T getData() const { return data; }
    RBTreeNode *getLeft() const { return left; }
    RBTreeNode *getRight() const { return right; }
    RBTreeNode *getParent() const { return parent; }
    void setLeft(RBTreeNode *node) { left = node; }
    void setRight(RBTreeNode *node) { right = node; }
    void setParent(RBTreeNode *node) { parent = node; }

private:
    Color color = Color::Red; // default color of node that inserted is red
    RBTreeNode *left = nullptr;
    RBTreeNode *right = nullptr;
    RBTreeNode *parent = nullptr;

    T data;
  };

  // Red-Black Tree
  template<typename T>
  class RBTree {
public:
    class iterator {};

    RBTree() = default;
    ~RBTree() { delete root; }

    void insert(T val) {
      if (!root) {
        root = new RBTreeNode<T>(val);
        root->setBlack();
        return;
      }
      RBTreeNode<T> *cur = root;
      
    }
    void remove(RBTreeNode<T> *node) {}

    RBTreeNode<T> *find(RBTreeNode<T> *node) {
      return nullptr;
    }
    RBTreeNode<T> *min() {
      return nullptr;
    }
    RBTreeNode<T> *max() {
      return nullptr;
    }

private:
    RBTreeNode<T> *root = nullptr;
  };
} // namespace hahaha::common::ds

#endif // HIAHIAHIA_RBTREE_H
