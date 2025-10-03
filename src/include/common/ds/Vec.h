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

#ifndef VEC_H
#define VEC_H
#include "common/defines/h3defs.h"
#include <algorithm>
#include <gtest/gtest-assertion-result.h>
#include <memory>


namespace hahaha::common::ds {
    template <class T, class Allocator = std::allocator<T>>
    class Vec {

    public:
        using valueTp        = T;
        using reference      = T&;
        using constReference = const T&;
        using pointer        = T*;
        using constPointer   = const T*;
        using iterator       = T*;
        using constIterator  = const T*;
        using sizeTp         = sizeT;
        using differenceTp   = ptrDiffT;

        Allocator _allocator;

        Vec() noexcept : _data(nullptr), _size(0), _capacity(0) {}

        explicit Vec(const sizeTp count) : _data(nullptr), _size(0), _capacity(0) {
            if (count > 0) {
            reserve(count);
                for (sizeT i = 0; i < count; ++i) {
                    std::allocator_traits<Allocator>::construct(_allocator, _data + i);
                }
                _size = count;
            }
        }

        Vec(std::initializer_list<T> init) : _data(nullptr), _size(0), _capacity(0) {
            reserve(init.size());
            _size = init.size();
            sizeT idx = 0;
            for (const auto& i : init) {
                std::allocator_traits<Allocator>::construct(_allocator, _data + idx, i);
                idx++;
            }
        }

        explicit Vec(Vec<T>::constIterator begin, Vec<T>::constIterator end) : _data(nullptr), _size(0), _capacity(0) {
            _size     = end - begin;
            _capacity = _size;
            if (_size > 0) {
                _data = _allocator.allocate(_capacity);
                for (auto i = begin; i < end; ++i) {
                    std::allocator_traits<Allocator>::construct(_allocator, _data + (i - begin), *i);
                }
            }
        }

        // Copy constructor
        Vec(const Vec& other) : _size(other._size), _capacity(other._capacity) {
            if (_capacity > 0) {
                _data = _allocator.allocate(_capacity);
                for (sizeT i = 0; i < _size; ++i) {
                    std::allocator_traits<Allocator>::construct(_allocator, _data + i, other._data[i]);
            }
            }
        }

        // Move constructor
        Vec(Vec&& other) noexcept : _data(other._data), _size(other._size), _capacity(other._capacity) {
            other._data     = nullptr;
            other._size     = 0;
            other._capacity = 0;
        }

        ~Vec() {
            for (sizeT i = 0; i < _size; ++i) {
                std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
            }
            _allocator.deallocate(_data, _capacity);
        }

        // Copy assignment operator
        Vec& operator=(const Vec& other) {
            if (this != &other) {
                // Destroy existing elements
                for (sizeT i = 0; i < _size; ++i) {
                    std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
                }
                // Deallocate old memory if needed
                if (_capacity < other._size) {
                    _allocator.deallocate(_data, _capacity);
                    _capacity = other._capacity;
                    _data     = _allocator.allocate(_capacity);
                }
                // Copy new elements
                _size = other._size;
                for (sizeT i = 0; i < _size; ++i) {
                    std::allocator_traits<Allocator>::construct(_allocator, _data + i, other._data[i]);
                }
            }
            return *this;
        }

        // Move assignment operator
        Vec& operator=(Vec&& other) noexcept {
            if (this != &other) {
                // Destroy existing elements
                for (sizeT i = 0; i < _size; ++i) {
                    std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
                }
                _allocator.deallocate(_data, _capacity);

                _data           = other._data;
                _size           = other._size;
                _capacity       = other._capacity;
                other._data     = nullptr;
                other._size     = 0;
                other._capacity = 0;
            }
            return *this;
        }

        void emplace_back(const T& value) {
            if (_size == _capacity) {
                reserve(_size == 0 ? 1 : _size * 2);
            }
            std::allocator_traits<Allocator>::construct(_allocator, _data + _size, value);
            _size++;
        }

        void push_back(const T& value) {
            emplace_back(value);
        }

        iterator begin() {
            return _data;
        }

        constIterator begin() const {
            return _data;
        }

        iterator end() {
            return _data + _size;
        }

        constIterator end() const {
            return _data + _size;
        }

        bool operator==(const Vec& vec) const {
            if (_size != vec._size) {
                return false;
            }

            for (sizeT i = 0; i < _size; i++) {
                if (!(_data[i] == vec._data[i])) { // Use ! for operator==
                    return false;
                }
            }

            return true;
        }

        [[nodiscard]] sizeT size() const {
            return _size;
        }

        [[nodiscard]] sizeT capacity() const {
            return _capacity;
        }

        T& operator[](const sizeTp index) {
            // No bounds checking for performance, like std::vector
            return _data[index];
        }

        const T& operator[](const sizeTp index) const {
            // No bounds checking for performance, like std::vector
            return _data[index];
        }

        void clear() {
            for (sizeT i = 0; i < _size; ++i) {
                std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
            }
            _size = 0;
        }

        void reserve(const sizeTp newCap) {
            if (newCap > _capacity) {
                T* new_data = _allocator.allocate(newCap);

                // Move existing elements to new storage
                for (sizeT i = 0; i < _size; ++i) {
                    std::allocator_traits<Allocator>::construct(_allocator, new_data + i, std::move(_data[i]));
                    std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
                }

                _allocator.deallocate(_data, _capacity);
                _data = new_data;
                _capacity = newCap;
            }
        }
        
        void shrink_to_fit() {
            if (_capacity > _size) {
                T* new_data = nullptr;
                if (_size > 0) {
                    new_data = _allocator.allocate(_size);
                    for (sizeT i = 0; i < _size; ++i) {
                        std::allocator_traits<Allocator>::construct(_allocator, new_data + i, std::move(_data[i]));
                        std::allocator_traits<Allocator>::destroy(_allocator, _data + i);
            }
                }
                _allocator.deallocate(_data, _capacity);
                _data = new_data;
                _capacity = _size;
            }
        }


        [[nodiscard]] bool empty() const {
            return _size == 0;
        }

        Vec<T> subVec(const sizeT start, const sizeT length) const {
            if (start + length > _size) {
                // Handle error: sub-vector out of bounds
                throw std::out_of_range("subVec: sub-vector range out of bounds");
            }
            Vec<T> res(length);
            for (sizeT i = 0; i < length; ++i) {
                res[i] = _data[start + i];
            }
            return res;
        }

    private:
        T* _data;
        sizeT _size;
        sizeT _capacity;
    };
} // namespace hahaha::common::ds


#endif // VEC_H
