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

#ifndef HASHMAP_H
#define HASHMAP_H

#include <vector>

#include "common/ds/Vector.h"

namespace hahaha::common::ds
{

template <typename Key, typename Value, typename Hash = std::hash<Key>> class HashMap
{
  public:
    using Pair = std::pair<const Key, Value>;

    HashMap(size_t initialCapacity = 8) : capacity_(initialCapacity), size_(0)
    {
        buckets_.resize(capacity_);
    }

    void insert(const Key& key, const Value& value)
    {
        if (size_ >= capacity_ / 2)
        {
            rehash(2 * capacity_);
        }

        size_t index = hash_(key) % capacity_;
        for (auto& pair : buckets_[index])
        {
            if (pair.first == key)
            {
                pair.second = value; // Update existing key
                return;
            }
        }

        buckets_[index].emplaceBack(key, value);
        size_++;
    }

    bool find(const Key& key, Value& value) const
    {
        size_t index = hash_(key) % capacity_;
        for (const auto& pair : buckets_[index])
        {
            if (pair.first == key)
            {
                value = pair.second;
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
            if (it->first == key)
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
    Vector<Vector<Pair>> buckets_;
    size_t capacity_;
    size_t size_;
    Hash hash_;

    void rehash(size_t newCapacity)
    {
        Vector<Vector<Pair>> newBuckets(newCapacity);
        for (const auto& bucket : buckets_)
        {
            for (const auto& pair : bucket)
            {
                size_t index = hash_(pair.first) % newCapacity;
                newBuckets[index].pushBack(pair);
            }
        }
        buckets_ = std::move(newBuckets);
        capacity_ = newCapacity;
    }
};

} // namespace hahaha::common::ds

#endif // HASHMAP_H
