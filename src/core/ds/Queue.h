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

#ifndef QUEUE_H
#define QUEUE_H

#include <initializer_list>
#include <stdexcept>

#include "Vector.h"

namespace hahaha::core::ds
{

template <class T, class Allocator = std::allocator<T>> class Queue
{
  public:
    // Member type aliases
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    // Default constructor
    Queue() noexcept : container_(), start_(0)
    {
    }

    // Constructor with initializer list
    Queue(std::initializer_list<T> init) : container_(init), start_(0)
    {
    }

    // Check if the queue is empty
    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

    // Get the number of elements in the queue
    [[nodiscard]] size_type size() const noexcept
    {
        return container_.size() - start_;
    }

    // Access the first element (modifiable)
    reference front()
    {
        if (empty())
        {
            throw std::out_of_range("queue is empty: cannot access front");
        }
        return container_[start_];
    }

    // Access the first element (const)
    const_reference front() const
    {
        if (empty())
        {
            throw std::out_of_range("queue is empty: cannot access front");
        }
        return container_[start_];
    }

    // Access the last element (modifiable)
    reference back()
    {
        if (empty())
        {
            throw std::out_of_range("queue is empty: cannot access back");
        }
        return container_[container_.size() - 1];
    }

    // Access the last element (const)
    const_reference back() const
    {
        if (empty())
        {
            throw std::out_of_range("queue is empty: cannot access back");
        }
        return container_[container_.size() - 1];
    }

    // Add an element to the end of the queue
    void push(const T& value)
    {
        container_.pushBack(value);
    }

    // Remove the first element from the queue
    void pop()
    {
        if (empty())
        {
            throw std::out_of_range("queue is empty: cannot pop");
        }
        ++start_;

        // Rebase to free memory if _start is more than half the container size
        if (start_ > container_.size() / 2)
        {
            rebase();
        }
    }

    // Clear all elements from the queue
    void clear() noexcept
    {
        start_ = container_.size(); // Effectively marks all elements as popped
    }

  private:
    Vector<T, Allocator> container_; // Underlying storage
    size_type start_;                // Index of the first active element

    // Rebase: move remaining elements to the start of the container to free
    // space
    void rebase()
    {
        const size_type new_size = size();
        Vector<T, Allocator> new_container(new_size); // Pre-reserve space

        // Copy active elements (from _start to end) to the new container
        for (size_type i = 0; i < new_size; ++i)
        {
            new_container.pushBack(container_[start_ + i]);
        }

        // Replace old container with the new one and reset _start
        container_ = std::move(new_container);
        start_ = 0;
    }
};

} // namespace hahaha::core::ds

#endif // QUEUE_H
