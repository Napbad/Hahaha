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
// Created by Napbad on 7/27/25.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "common/Error.h"
#include "common/Res.h"
#include "common/defines/h3defs.h"
#include "common/ds/Vec.h"

namespace hahaha::ml {

  using namespace hahaha::common;

  class TensorErr final : public BaseErr {
public:
    TensorErr() = default;
    explicit TensorErr(const char *msg) : BaseErr(msg) {}
  };

  template<typename T>
  class Tensor {
public:
    using valueType = T;

    Tensor() = default;

    Tensor(std::initializer_list<sizeT> shape) : _shape(shape), _data(computeSize(shape)) {}

    explicit Tensor(const ds::Vec<sizeT> &shape) : _shape(shape), _data(computeSize(shape)) {}

    // Access shape
    [[nodiscard]] const ds::Vec<sizeT> &shape() const { return _shape; }
    [[nodiscard]] sizeT size() const { return _data.size(); }

    // Index calculation (flattened)
    [[nodiscard]] Res<sizeT, BaseErr> index(const ds::Vec<sizeT> &indices) const {
      SetRetT(sizeT, BaseErr)

              if (indices.size() != _shape.size()) Err(BaseErr("Incorrect number of indices"));
      sizeT idx = 0;
      sizeT stride = 1;
      for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        if (indices[i] >= _shape[i])
          Err(BaseErr("Index out of bounds"));
        idx += indices[i] * stride;
        stride *= _shape[i];
      }
      Ok(idx);
    }

    // Element access
    valueType &operator()(const ds::Vec<sizeT> &indices) { return _data[index(indices).unwrap()]; }
    const valueType &operator()(const ds::Vec<sizeT> &indices) const { return _data[index(indices).unwrap()]; }

    // Fill tensor with value
    void fill(const valueType v) const { std::ranges::fill(_data, v); }

    // Element-wise addition
    Tensor operator+(const Tensor &other) const {
      checkShape(other);
      Tensor result(_shape);
      for (sizeT i = 0; i < size(); ++i)
        result._data[i] = _data[i] + other._data[i];
      return result;
    }

    // Element-wise multiplication
    Tensor operator*(const Tensor &other) const {
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
      for (const auto &v: _data)
        std::cout << v << " ";
      std::cout << "\n";
    }

    // Static factory methods
    static Tensor fromVector(const ds::Vec<T> &vec) {
      Tensor tensor({vec.size()});
      for (sizeT i = 0; i < vec.size(); ++i) {
        tensor._data[i] = vec[i];
      }
      return tensor;
    }

    Res<void, TensorErr> copy(const Tensor other) {
      SetRetT(void, TensorErr) if (other.shape() != _shape) {
        Err("can not copy value from a tensor with different shape")
      }

      for (int i = 0; i < other.size(); ++i) {
        if (auto val = dynamic_cast<Tensor *>(other._data[i])) {
          dynamic_cast<Tensor *>(_data[i])->copy(*val);
        } else {
          val = other._data[i];
        }
      }
      Ok()
    }

    [[nodiscard]] sizeT dim() const { return shape().size(); }

private:
    ds::Vec<sizeT> _shape;
    ds::Vec<valueType> _data;

    static sizeT computeSize(const ds::Vec<sizeT> &shape) {
      return std::accumulate(shape.begin(), shape.end(), sizeT{1}, std::multiplies<>());
    }

    void checkShape(const Tensor &other) const {
      if (_shape != other._shape)
        throw std::runtime_error("Shape mismatch");
    }
  };

} // namespace hahaha::ml
#endif // TENSOR_H
