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
#include "common/Res.h"
#include "common/defines/h3defs.h"
#include "common/ds/Vec.h"
#include "common/error.h"
#include <initializer_list>
#include <iostream>
#include <numeric>

namespace hahaha::ml {

    using namespace hahaha::common;

    class TensorErr final : public BaseErr {
    public:
        TensorErr() = default;
        explicit TensorErr(const char* msg) : BaseErr(msg) {}
        explicit TensorErr(const Str& msg) : BaseErr(msg) {}
    };

    template <typename T>
    class Tensor {
    public:
        using valueType = T;

        Tensor() = default;

        Tensor(const std::initializer_list<sizeT> shape) : _shape(shape), _data(computeSize(shape)) {}

        explicit Tensor(const ds::Vec<sizeT>& shape) : _shape(shape), _data(computeSize(shape)) {}

        // Constructor for a 0-dimensional tensor (scalar)
        explicit Tensor(const std::initializer_list<sizeT> shape, std::initializer_list<T> data)
            : _shape(shape), _data(data) {}

        // Access shape
        [[nodiscard]] const ds::Vec<sizeT>& shape() const {
            return _shape;
        }
        [[nodiscard]] sizeT size() const {
            return _data.size();
        }

        // Index calculation (flattened)
        [[nodiscard]] Res<sizeT, BaseErr> index(const std::initializer_list<sizeT> indices) const {
            SetRetT(sizeT, BaseErr)

                if (indices.size() != _shape.size()) Err(BaseErr("Incorrect number of indices"));
            sizeT idx    = 0;
            sizeT stride = 1;
            for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
                if (indices.begin()[i] >= _shape[i]) {
                    Err(BaseErr("Index out of bounds"));
                }
                idx += indices.begin()[i] * stride;
                stride *= _shape[i];
            }
            Ok(idx);
        }

        // Element access
        valueType& operator()(const ds::Vec<sizeT>& indices) {
            return _data[index(indices).unwrap()];
        }
        const valueType& operator()(const ds::Vec<sizeT>& indices) const {
            return _data[index(indices).unwrap()];
        }

        // Fill tensor with value
        void fill(const valueType v) {
            std::ranges::fill(_data, v);
        }

        // Element-wise addition
        Tensor operator+(const Tensor& other) const {
            checkShape(other);
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] + other._data[i];
            }
            return result;
        }

        // Element-wise multiplication
        Tensor operator*(const Tensor& other) const {
            checkShape(other);
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] * other._data[i];
            }
            return result;
        }

        // Scalar operations
        Tensor operator*(valueType scalar) const {
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] * scalar;
            }
            return result;
        }

        T dot(const Tensor& other) const {
            checkShape(other);
            T result = 0;
            for (sizeT i = 0; i < size(); ++i) {
                result += _data[i] * other._data[i];
            }
            return result;
        }

        // Print
        void printFlat() const {
            for (const auto& v : _data) {
                std::cout << v << " ";
            }
            std::cout << "\n";
        }

        T &first() {
            return _data[0];
        }

        // Static factory methods
        static Tensor fromVector(const ds::Vec<T>& vec) {
            Tensor tensor({vec.size()});
            for (sizeT i = 0; i < vec.size(); ++i) {
                tensor._data[i] = vec[i];
            }
            return tensor;
        }

        Res<void, TensorErr> copy(const Tensor& other) {
            SetRetT(void, TensorErr) if (other.shape() != _shape) {
                Err(TensorErr("Cannot copy value from a tensor with different shape"))
            }
            for (sizeT i = 0; i < other.size(); ++i) {
                _data[i] = other._data[i];
            }
            Ok()
        }

        Res<void, TensorErr> copy(const ds::Vec<T>& other) {
            SetRetT(void, TensorErr) if (other.size() != this->size()) {
                Err(TensorErr("Cannot copy value from a tensor with different shape"))
            }
            for (sizeT i = 0; i < other.size(); ++i) {
                _data[i] = other[i];
            }
            Ok()
        }

        [[nodiscard]] sizeT dim() const {
            return shape().size();
        }

        // Element-wise subtraction
        Tensor operator-(const Tensor& other) const {
            checkShape(other);
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] - other._data[i];
            }
            return result;
        }

        // Element-wise division
        Tensor operator/(const Tensor& other) const {
            checkShape(other);
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                // Add check for division by zero if T is a floating-point type
                if constexpr (std::is_floating_point_v<T>) {
                    if (other._data[i] == static_cast<T>(0)) {
                        throw std::runtime_error("Division by zero");
                    }
                }
                result._data[i] = _data[i] / other._data[i];
            }
            return result;
        }

        // Scalar operations (subtraction and division)
        Tensor operator-(valueType scalar) const {
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] - scalar;
            }
            return result;
        }

        Tensor operator/(valueType scalar) const {
            if constexpr (std::is_floating_point_v<T>) {
                if (scalar == static_cast<T>(0)) {
                    throw std::runtime_error("Division by zero by scalar");
                }
            }
            Tensor result(_shape);
            for (sizeT i = 0; i < size(); ++i) {
                result._data[i] = _data[i] / scalar;
            }
            return result;
        }

        // Compound assignment operators
        Tensor& operator+=(const Tensor& other) {
            checkShape(other);
            for (sizeT i = 0; i < size(); ++i) {
                _data[i] += other._data[i];
            }
            return *this;
        }

        Tensor& operator-=(const Tensor& other) {
            checkShape(other);
            for (sizeT i = 0; i < size(); ++i) {
                _data[i] -= other._data[i];
            }
            return *this;
        }

        Tensor& operator*=(const Tensor& other) {
            checkShape(other);
            for (sizeT i = 0; i < size(); ++i) {
                _data[i] *= other._data[i];
            }
            return *this;
        }

        Tensor& operator/=(const Tensor& other) {
            checkShape(other);
            for (sizeT i = 0; i < size(); ++i) {
                if constexpr (std::is_floating_point_v<T>) {
                    if (other._data[i] == static_cast<T>(0)) {
                        throw std::runtime_error("Division by zero");
                    }
                }
                _data[i] /= other._data[i];
            }
            return *this;
        }

        [[nodiscard]] auto begin() const {
            return _data.begin();
        }
        [[nodiscard]] auto end() const {
            return _data.end();
        }

        [[nodiscard]] auto rawData() const {
            return _data;
        }

        [[nodiscard]] bool empty() const {
            return _data.empty();
        }

        Res<Tensor, IndexOutOfBoundError> at(const std::initializer_list<sizeT> indices) const {
            SetRetT(Tensor, IndexOutOfBoundError) if (indices.size() > dim()) {
                Err("Too many indices for at() method");
            }
            if (indices.size() == 0 && dim() != 0) {
                Err("Cannot access scalar tensor with empty indices")
            }
            // calculate start index and tensor size
            auto idxRes = index(indices);
            if (idxRes.isErr()) {
                Err(IndexOutOfBoundError(idxRes.unwrapErr().message()));
            }

            sizeT start  = idxRes.unwrap();
            sizeT length = 1;

            if (indices.size() == dim()) {
                Ok(Tensor({0}, {_data[start]}))
            }

            ds::Vec<sizeT> newShape;
            for (sizeT i = indices.size(); i < _shape.size(); ++i) {
                newShape.push_back(i);
                length *= _shape[i];
            }

            if (start + length > _data.size()) {
                Err(IndexOutOfBoundError("Calculated range out of bounds"));
            }

            Tensor res(newShape);
            // res.replaceSelf(_data.begin() + start, _data.begin() + start + length);
            res.copy(_data.subVec(start, length));
            Ok(res);
        }

        Res<void, IndexOutOfBoundError> set(const std::initializer_list<sizeT> indices, T value) {
            SetRetT(void, IndexOutOfBoundError) if (indices.size() > dim()) {
                Err("Too many indices for set() method");
            }
            if (indices.size() == 0 && dim() != 0) {
                Err("Cannot access scalar tensor with empty indices")
            }
            // calculate start index and tensor size
            auto idxRes = index(indices);
            if (idxRes.isErr()) {
                Err(IndexOutOfBoundError(idxRes.unwrapErr().message()));
            }

            _data[idxRes.unwrap()] = value;
            Ok()
        }

    private:
        ds::Vec<sizeT> _shape;
        ds::Vec<valueType> _data;

        static sizeT computeSize(const ds::Vec<sizeT>& shape) {
            return std::accumulate(shape.begin(), shape.end(), sizeT{1}, std::multiplies<>());
        }

        void checkShape(const Tensor& other) const {
            if (_shape != other._shape) {
                throw std::runtime_error("Shape mismatch");
            }
        }
    };

} // namespace hahaha::ml
#endif // TENSOR_H
