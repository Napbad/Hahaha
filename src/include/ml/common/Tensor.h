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
// Created by root on 7/27/25.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "include/common/Error.h"
#include "include/common/Res.h"
#include "include/common/defines/h3defs.h"
#include "include/common/ds/vec.h"

namespace hiahiahia {


class Tensor {
public:
    using valueType = float; // You can make this a template parameter for other types

    Tensor() = default;

    Tensor(std::initializer_list<hiahiahia::sizeT> shape)
        : _shape(shape), _data(computeSize(shape)) {}

    explicit Tensor(const ds::vec<sizeT>& shape)
        : _shape(shape), _data(computeSize(shape)) {}

    // Access shape
    [[nodiscard]] const ds::vec<sizeT>& shape() const { return _shape; }
    [[nodiscard]] sizeT size() const { return _data.size(); }

    // Index calculation (flattened)
  [[nodiscard]] Res<sizeT, err> index(const ds::vec<sizeT>& indices) const {
      if (indices.size() != _shape.size())
        return err(std::make_unique<err>("Incorrect number of indices"));
      sizeT idx = 0;
      sizeT stride = 1;
      for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        if (indices[i] >= _shape[i])
          return err(std::make_unique<err>("Index out of bounds"));
        idx += indices[i] * stride;
        stride *= _shape[i];
      }
      return Ok(idx);
    }

  // Element access
    valueType& operator()(const ds::vec<sizeT>& indices) {
        return _data[index(indices)];
    }
    const valueType& operator()(const ds::vec<sizeT>& indices) const {
        return _data[index(indices)];
    }

    // Fill tensor with value
    void fill(const valueType v) const {
        std::ranges::fill(_data, v);
    }

    // Element-wise addition
    Tensor operator+(const Tensor& other) const {
        checkShape(other);
        Tensor result(_shape);
        for (sizeT i = 0; i < size(); ++i)
            result._data[i] = _data[i] + other._data[i];
        return result;
    }

    // Element-wise multiplication
    Tensor operator*(const Tensor& other) const {
        checkShape(other);
        Tensor result(_shape);
        for (sizeT i = 0; i < size(); ++i)
            result._data[i] = _data[i] * other._data[i];
        return result;
    }

    // Scalar operations
    Tensor operator*(valueType scalar) const {
        Tensor result(_shape);
        for (sizeT i = 0; i < size(); ++i)
            result._data[i] = _data[i] * scalar;
        return result;
    }

    // Print
    void printFlat() const {
        for (const auto & v : _data) std::cout << v << " ";
        std::cout << "\n";
    }

private:
    ds::vec<sizeT> _shape;
    ds::vec<valueType> _data;

    static sizeT computeSize(const ds::vec<sizeT>& shape) {
        return std::accumulate(shape.begin(), shape.end(), sizeT{1}, std::multiplies<>());
    }

    void checkShape(const Tensor& other) const {
        if (_shape != other._shape)
            throw std::runtime_error("Shape mismatch");
    }
};

} // namespace hiahiahia
#endif //TENSOR_H
