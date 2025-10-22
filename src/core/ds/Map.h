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

#ifndef MAP_H
#define MAP_H

#include <stdexcept>
#include <utility>
#include <vector>

#include "rbTree.h"

namespace hahaha::core::ds
{

// Custom pair that supports comparison by key
template <typename Key, typename T> struct KeyValuePair
{
    Key first;
    T second;

    KeyValuePair(const Key& k, const T& v) : first(k), second(v)
    {
    }
    explicit KeyValuePair(const std::pair<Key, T>& p)
        : first(p.first), second(p.second)
    {
    }

    bool operator<(const KeyValuePair& other) const
    {
        return first < other.first;
    }

    bool operator>(const KeyValuePair& other) const
    {
        return first > other.first;
    }

    bool operator==(const KeyValuePair& other) const
    {
        return first == other.first;
    }
};

template <typename Key, typename T, typename Compare = std::less<Key>> class Map
{
  public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using size_type = size_t;
    using kvpair = KeyValuePair<Key, T>;

    // Simple iterator for range-based for loops
    class iterator
    {
      public:
        iterator(const std::vector<kvpair*>* vec, size_t index)
            : vec_(vec), index_(index)
        {
        }

        std::pair<const Key&, T&> operator*() const
        {
            return {(*vec_)[index_]->first, (*vec_)[index_]->second};
        }

        iterator& operator++()
        {
            ++index_;
            return *this;
        }

        bool operator!=(const iterator& other) const
        {
            return index_ != other.index_;
        }

      private:
        const std::vector<kvpair*>* vec_;
        size_t index_;
    };

    class const_iterator
    {
      public:
        const_iterator(const std::vector<kvpair*>* vec, size_t index)
            : vec_(vec), index_(index)
        {
        }

        std::pair<const Key&, const T&> operator*() const
        {
            return {(*vec_)[index_]->first, (*vec_)[index_]->second};
        }

        const_iterator& operator++()
        {
            ++index_;
            return *this;
        }

        bool operator!=(const const_iterator& other) const
        {
            return index_ != other.index_;
        }

      private:
        const std::vector<kvpair*>* vec_;
        size_t index_;
    };

    Map() = default;
    ~Map() = default;

    // Iterators
    iterator begin()
    {
        updateIteratorCache();
        return iterator(&iterCache_, 0);
    }

    iterator end()
    {
        updateIteratorCache();
        return iterator(&iterCache_, iterCache_.size());
    }

    const_iterator begin() const
    {
        updateIteratorCache();
        return const_iterator(&iterCache_, 0);
    }

    const_iterator end() const
    {
        updateIteratorCache();
        return const_iterator(&iterCache_, iterCache_.size());
    }

    // Element access
    T& operator[](const Key& key)
    {
        kvpair searchPair(key, T());

        if (auto node = tree_.find(searchPair))
        {
            // Return reference to the actual stored second value
            return node->getDataRef().second;
        }
        else
        {
            // Insert with default value
            tree_.insert(searchPair);
            auto inserted = tree_.find(searchPair);
            invalidateIteratorCache();
            return inserted->getDataRef().second;
        }
    }

    T& at(const Key& key)
    {
        kvpair searchPair(key, T());
        auto node = tree_.find(searchPair);

        if (!node)
            throw std::out_of_range("Map::at: key not found");

        return node->getDataRef().second;
    }

    const T& at(const Key& key) const
    {
        kvpair searchPair(key, T());
        auto node = tree_.find(searchPair);

        if (!node)
            throw std::out_of_range("Map::at: key not found");

        return node->getDataConstRef().second;
    }

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
    std::pair<bool, T*> insert(const Key& key, const T& value)
    {
        kvpair newPair(key, value);
        auto existing = tree_.find(newPair);

        if (existing)
        {
            return {false, &existing->getDataRef().second};
        }

        tree_.insert(newPair);
        auto inserted = tree_.find(newPair);
        invalidateIteratorCache();
        return {true, &inserted->getDataRef().second};
    }

    std::pair<bool, T*> insert(const std::pair<Key, T>& pair)
    {
        return insert(pair.first, pair.second);
    }

    template <typename... Args>
    std::pair<bool, T*> emplace(const Key& key, Args&&... args)
    {
        kvpair searchPair(key, T());
        auto existing = tree_.find(searchPair);

        if (existing)
        {
            return {false, &existing->getDataRef().second};
        }

        T value(std::forward<Args>(args)...);
        kvpair newPair(key, value);
        tree_.insert(newPair);
        auto inserted = tree_.find(newPair);
        invalidateIteratorCache();
        return {true, &inserted->getDataRef().second};
    }

    size_type erase(const Key& key)
    {
        kvpair searchPair(key, T());
        auto node = tree_.find(searchPair);

        if (!node)
            return 0;

        tree_.remove(searchPair);
        invalidateIteratorCache();
        return 1;
    }

    void clear()
    {
        while (tree_.getRoot())
        {
            tree_.remove(tree_.getRoot()->getData());
        }
        invalidateIteratorCache();
    }

    // Lookup
    size_type count(const Key& key) const
    {
        kvpair searchPair(key, T());
        return tree_.find(searchPair) != nullptr ? 1 : 0;
    }

    bool contains(const Key& key) const
    {
        return count(key) > 0;
    }

    T* find(const Key& key)
    {
        kvpair searchPair(key, T());
        auto node = tree_.find(searchPair);
        return node ? &node->getDataRef().second : nullptr;
    }

    const T* find(const Key& key) const
    {
        kvpair searchPair(key, T());
        auto node = tree_.find(searchPair);
        return node ? &node->getDataConstRef().second : nullptr;
    }

    // Additional utility methods
    std::vector<Key> keys() const
    {
        std::vector<Key> result;
        collectKeys(tree_.getRoot(), result);
        return result;
    }

    std::vector<T> values() const
    {
        std::vector<T> result;
        collectValues(tree_.getRoot(), result);
        return result;
    }

    std::vector<std::pair<Key, T>> entries() const
    {
        std::vector<std::pair<Key, T>> result;
        collectEntries(tree_.getRoot(), result);
        return result;
    }

  private:
    RBTree<kvpair> tree_;
    mutable std::vector<kvpair*> iterCache_;
    mutable bool iterCacheValid_ = false;

    void updateIteratorCache() const
    {
        if (!iterCacheValid_)
        {
            iterCache_.clear();
            collectNodePointers(tree_.getRoot(), iterCache_);
            iterCacheValid_ = true;
        }
    }

    void invalidateIteratorCache()
    {
        iterCacheValid_ = false;
    }

    void collectNodePointers(RBTreeNode<kvpair>* node,
                             std::vector<kvpair*>& pointers) const
    {
        if (!node)
            return;
        collectNodePointers(node->getLeft(), pointers);
        pointers.push_back(
            &const_cast<RBTreeNode<kvpair>*>(node)->getDataRef());
        collectNodePointers(node->getRight(), pointers);
    }

    size_type countNodes(RBTreeNode<kvpair>* node) const
    {
        if (!node)
            return 0;
        return 1 + countNodes(node->getLeft()) + countNodes(node->getRight());
    }

    void collectKeys(RBTreeNode<kvpair>* node, std::vector<Key>& keys) const
    {
        if (!node)
            return;
        collectKeys(node->getLeft(), keys);
        keys.push_back(node->getDataConstRef().first);
        collectKeys(node->getRight(), keys);
    }

    void collectValues(RBTreeNode<kvpair>* node, std::vector<T>& values) const
    {
        if (!node)
            return;
        collectValues(node->getLeft(), values);
        values.push_back(node->getDataConstRef().second);
        collectValues(node->getRight(), values);
    }

    void collectEntries(RBTreeNode<kvpair>* node,
                        std::vector<std::pair<Key, T>>& entries) const
    {
        if (!node)
            return;
        collectEntries(node->getLeft(), entries);
        const auto& data = node->getDataConstRef();
        entries.push_back({data.first, data.second});
        collectEntries(node->getRight(), entries);
    }
};

} // namespace hahaha::core::ds

#endif // MAP_H
