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

#ifndef HASHSET_H
#define HASHSET_H

#include <vector>

#include "core/ds/Vector.h"

namespace hahaha::core::ds
{
template <typename Key, typename Hash = std::hash<Key>> class HashSet
{
  public:
    HashSet(size_t initialCapacity = 8) : capacity_(initialCapacity), size_(0)
    {
        buckets_.resize(capacity_);
    }

    void insert(const Key& key)
    {
        if (size_ >= capacity_ / 2)
        {
            rehash(2 * capacity_);
        }

        size_t index = hash_(key) % capacity_;
        for (const auto& k : buckets_[index])
        {
            if (k == key)
            {
                return; // Key already exists
            }
        }

        buckets_[index].pushBack(key);
        size_++;
    }

    bool contains(const Key& key) const
    {
        size_t index = hash_(key) % capacity_;
        for (const auto& k : buckets_[index])
        {
            if (k == key)
            {
                return true;
            }
        }
        return false;
    }

    bool erase(const Key& key)
    {
        size_t index = hash_(key) % capacity_;
        auto& bucket = buckets_[index];
        for (auto it = bucket.begin(); it != bucket.end(); ++it)
        {
            if (*it == key)
            {
                bucket.erase(it);
                size_--;
                return true;
            }
        }
        return false;
    }

    size_t size() const
    {
        return size_;
    }

  private:
    Vector<Vector<Key>> buckets_;
    size_t capacity_;
    size_t size_;
    Hash hash_;

    void rehash(size_t newCapacity)
    {
        Vector<Vector<Key>> newBuckets(newCapacity);
        for (const auto& bucket : buckets_)
        {
            for (const auto& key : bucket)
            {
                size_t index = hash_(key) % newCapacity;
                newBuckets[index].pushBack(key);
            }
        }
        buckets_ = std::move(newBuckets);
        capacity_ = newCapacity;
    }
};

} // namespace hahaha::core::ds

#endif // HASHSET_H
