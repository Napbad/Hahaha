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
// Created by napbad on 7/12/25.
//

#ifndef TENSOR_H
#define TENSOR_H

#include "TensorShape.h"
#include "TensorStride.h"

namespace hiahiahia {

    template<typename T>
    class Tensor {
    public:

        Tensor(T* data, const uint shapeSize, const uint* shape):
        _shape(shape, shapeSize), _stride(shapeSize) {
            this->_data = data;
            this->_stride.calculateStride(shape, shapeSize);
            this->_require_grad = false;
        }

        explicit Tensor(const Tensor * other): _shape(other->_shape), _stride(other->_stride)
        {
            this->_data = new T[this->_shape.getTotalSize()];

            for (uint i = 0; i < this->_shape.getTotalSize(); ++i)
                this._data[i] = other->_data[i];

            this->_require_grad = false;
        }

        explicit Tensor(const TensorShape& shape): _shape(shape), _stride(shape)
        {
            this->_data = new T[this->_shape.getTotalSize()];

            this->_require_grad = false;
        }

        void permute(TensorShape *newShape)
        {

        }
        Tensor operator-(const Tensor& other) const
        {
            if (this->_shape != other._shape)
                return Tensor();

            auto res = Tensor(_shape);
            for (uint i = 0; i < this->_shape.getTotalSize(); ++i)
                res._data[i] = this->_data[i] - other._data[i];
            return res;
        }

        Tensor operator+(const Tensor& other) const
        {
            if (this->_shape != other._shape)
                return Tensor();

            auto res = Tensor(_shape);
            for (uint i = 0; i < this->_shape.getTotalSize(); ++i)
                res._data[i] = this->_data[i] + other._data[i];
            return res;
        }

        Tensor operator*(const Tensor& other) const
        {
            if (this->_shape != other._shape)
                return Tensor();

            auto res = Tensor(_shape);
            for (uint i = 0; i < this->_shape.getTotalSize(); ++i)
                res._data[i] = this->_data[i] * other._data[i];
            return res;
        }

        ~Tensor() {
            delete _data;
        }
    private:
        TensorShape _shape;
        TensorStride _stride;
        bool _require_grad;

        T* _data = nullptr;
    };

}

#endif //TENSOR_H
