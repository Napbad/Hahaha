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
#include <algorithm>
#include <memory>

#include "common/defines/h3defs.h"


namespace hiahiahia::ds {
  template<class T, class Allocator = std::allocator<T>>
  class Vec {

public:

    using valueTp = T;
    using reference = T&;
    using constReference = const T&;
    using pointer = T*;
    using constPointer = const T*;
    using iterator = T*;
    using constIterator = const T*;
    using sizeTp = sizeT;
    using differenceTp = ptrDiffT;

    Vec ()  noexcept : _data(nullptr), _size(0), _capacity(0) {

    }

    explicit Vec(const sizeTp count): _data(nullptr), _size(0), _capacity(count) {
      reserve(count);
    }
    Vec(std::initializer_list<T> shape) {
      _data = new T[shape.size()];
      _size = shape.size();
      _capacity = shape.size();
      sizeT idx = 0;
      for (auto i : shape) {
        _data[idx++] = i;
      }
    }

    ~Vec() {
      delete []_data;
    }

    void emplace_back(const T & value) {
      if (_size == _capacity) {
        reserve(_size * 2);
      }
      _data[_size++] = value;
    }

    void push_back(const T & value) {
      emplace_back(value);
    }

    iterator begin() const {
      return _data;
    }

    iterator end() const {
      return &_data[_size - 1];
    }
    bool operator==(const Vec & vec) const {
      if (_size != vec._size) {
        return false;
      }

      for (int i = 0; i < _size; i++) {
        if (_data[i] != vec._data[i]) {
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

    T & operator[](const sizeTp index) {
      return _data[index];
    }

    T & operator[](const sizeTp index) const {
      return _data[index];
    }

    void clear() {
      delete []_data;
      _size = 0;
      _capacity = 0;
    }

    void reserve(const sizeTp newCap) {
      auto newData = static_cast<pointer>(operator new(newCap * sizeof(T)));
      for (auto i = 0; i < _size; ++i) {
        new (newData + i) T(std::move(_data[i]));
        _data[i].~T();
      }

      delete [] _data;
      _capacity = newCap;
      _data = newData;
    }
private:
    T *_data;
    sizeT _size;
    sizeT _capacity;

  };
} // namespace hiahiahia::ds


#endif // VEC_H
