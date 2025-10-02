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

#include "Vec.h"
#include <initializer_list>
#include <stdexcept>

namespace hahaha::common::ds {

    template <class T, class Allocator = std::allocator<T>>
    class queue {
    public:
        // Member type aliases
        using value_type      = T;
        using reference       = T&;
        using const_reference = const T&;
        using pointer         = T*;
        using const_pointer   = const T*;
        using size_type       = size_t;

        // Default constructor
        queue() noexcept : _container(), _start(0) {}

        // Constructor with initializer list
        queue(std::initializer_list<T> init) : _container(init), _start(0) {}

        // Check if the queue is empty
        [[nodiscard]] bool empty() const noexcept {
            return size() == 0;
        }

        // Get the number of elements in the queue
        [[nodiscard]] size_type size() const noexcept {
            return _container.size() - _start;
        }

        // Access the first element (modifiable)
        reference front() {
            if (empty()) {
                throw std::out_of_range("queue is empty: cannot access front");
            }
            return _container[_start];
        }

        // Access the first element (const)
        const_reference front() const {
            if (empty()) {
                throw std::out_of_range("queue is empty: cannot access front");
            }
            return _container[_start];
        }

        // Access the last element (modifiable)
        reference back() {
            if (empty()) {
                throw std::out_of_range("queue is empty: cannot access back");
            }
            return _container[_container.size() - 1];
        }

        // Access the last element (const)
        const_reference back() const {
            if (empty()) {
                throw std::out_of_range("queue is empty: cannot access back");
            }
            return _container[_container.size() - 1];
        }

        // Add an element to the end of the queue
        void push(const T& value) {
            _container.push_back(value);
        }

        // Remove the first element from the queue
        void pop() {
            if (empty()) {
                throw std::out_of_range("queue is empty: cannot pop");
            }
            ++_start;

            // Rebase to free memory if _start is more than half the container size
            if (_start > _container.size() / 2) {
                rebase();
            }
        }

        // Clear all elements from the queue
        void clear() noexcept {
            _start = _container.size(); // Effectively marks all elements as popped
        }

    private:
        Vec<T, Allocator> _container; // Underlying storage
        size_type _start; // Index of the first active element

        // Rebase: move remaining elements to the start of the container to free space
        void rebase() {
            const size_type new_size = size();
            Vec<T, Allocator> new_container(new_size); // Pre-reserve space

            // Copy active elements (from _start to end) to the new container
            for (size_type i = 0; i < new_size; ++i) {
                new_container.push_back(_container[_start + i]);
            }

            // Replace old container with the new one and reset _start
            _container = std::move(new_container);
            _start     = 0;
        }
    };

} // namespace hahaha::common::ds

#endif // QUEUE_H
