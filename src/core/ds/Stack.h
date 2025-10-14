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

#ifndef STACK_H
#define STACK_H
#include "Vector.h"

namespace hahaha::core::ds
{
template <typename T> class Stack
{
  public:
    // Member type aliases
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    Stack(std::initializer_list<T> init) : data_(init)
    {
    }

    Stack() = default;

    void pop()
    {
        if (empty())
        {
            throw std::runtime_error("Cannot pop from an empty stack.");
        }
        data_.popBack();
    }

    [[nodiscard]] reference top()
    {
        if (empty())
        {
            throw std::runtime_error("Cannot get top of an empty stack.");
        }
        return data_.back();
    }

    [[nodiscard]] const_reference top() const
    {
        if (empty())
        {
            throw std::runtime_error("Cannot get top of an empty stack.");
        }
        return data_.back();
    }

    void push(const T& item)
    {
        data_.pushBack(item);
    }

    [[nodiscard]] bool empty() const
    {
        return data_.empty();
    }

    [[nodiscard]] sizeT size() const
    {
        return data_.size();
    }

    [[nodiscard]] sizeT capacity() const
    {
        return data_.capacity();
    }

  private:
    Vector<T> data_;
};
} // namespace hahaha::core::ds

#endif // STACK_H
