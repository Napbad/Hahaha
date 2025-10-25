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
// Created by Napbad on 7/26/25.
//

#ifndef SET_H
#define SET_H

#include <stdexcept>
#include <vector>

#include "RBTree.h"

namespace hahaha::core::ds
{

template <typename T, typename Compare = std::less<T>> class Set
{
  public:
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;

    Set() = default;
    ~Set() = default;

    // Capacity
    [[nodiscard]] bool empty() const
    {
        return tree_.getRoot() == nullptr;
    }

    [[nodiscard]] size_type size() const
    {
        return countNodes(tree_.getRoot());
    }

    // Modifiers
    bool insert(const T& value)
    {
        auto existing = tree_.find(value);
        if (existing)
            return false;

        tree_.insert(value);
        return true;
    }

    template <typename... Args> bool emplace(Args&&... args)
    {
        T value(std::forward<Args>(args)...);
        return insert(value);
    }

    size_type erase(const T& value)
    {
        auto node = tree_.find(value);
        if (!node)
            return 0;

        tree_.remove(value);
        return 1;
    }

    void clear()
    {
        while (tree_.getRoot())
        {
            tree_.remove(tree_.getRoot()->getData());
        }
    }

    // Lookup
    size_type count(const T& value) const
    {
        return tree_.find(value) != nullptr ? 1 : 0;
    }

    bool contains(const T& value) const
    {
        return tree_.find(value) != nullptr;
    }

    const T* find(const T& value) const
    {
        auto node = tree_.find(value);
        return node ? &node->getDataConstRef() : nullptr;
    }

    // Min/Max
    const T* min() const
    {
        auto node = tree_.min();
        return node ? &node->getDataConstRef() : nullptr;
    }

    const T* max() const
    {
        auto node = tree_.max();
        return node ? &node->getDataConstRef() : nullptr;
    }

    // Set operations
    std::vector<T> toVector() const
    {
        std::vector<T> result;
        collectValues(tree_.getRoot(), result);
        return result;
    }

    // Union: returns a new set containing all elements from both sets
    Set unionWith(const Set& other) const
    {
        Set result;
        auto thisValues = toVector();
        auto otherValues = other.toVector();

        for (const auto& val : thisValues)
        {
            result.insert(val);
        }
        for (const auto& val : otherValues)
        {
            result.insert(val);
        }
        return result;
    }

    // Intersection: returns a new set containing only common elements
    Set intersectWith(const Set& other) const
    {
        Set result;
        auto thisValues = toVector();
        for (const auto& val : thisValues)
        {
            if (other.contains(val))
            {
                result.insert(val);
            }
        }
        return result;
    }

    // Difference: returns a new set with elements in this set but not in other
    Set difference(const Set& other) const
    {
        Set result;
        auto thisValues = toVector();
        for (const auto& val : thisValues)
        {
            if (!other.contains(val))
            {
                result.insert(val);
            }
        }
        return result;
    }

    // Subset check: returns true if this set is a subset of other
    bool isSubsetOf(const Set& other) const
    {
        auto thisValues = toVector();
        for (const auto& val : thisValues)
        {
            if (!other.contains(val))
                return false;
        }
        return true;
    }

    // Superset check: returns true if this set is a superset of other
    bool isSupersetOf(const Set& other) const
    {
        return other.isSubsetOf(*this);
    }

  private:
    RBTree<T> tree_;

    size_type countNodes(RBTreeNode<T>* node) const
    {
        if (!node)
            return 0;
        return 1 + countNodes(node->getLeft()) + countNodes(node->getRight());
    }

    void collectValues(RBTreeNode<T>* node, std::vector<T>& values) const
    {
        if (!node)
            return;
        collectValues(node->getLeft(), values);
        values.push_back(node->getDataConstRef());
        collectValues(node->getRight(), values);
    }
};

} // namespace hahaha::core::ds

#endif // SET_H
