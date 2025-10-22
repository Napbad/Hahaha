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
// Created by root on 10/21/25.
//

#ifndef HAHAHA_TENSORVAR_H
#define HAHAHA_TENSORVAR_H
#include "TensorPtr.h"
#include "core/compute/autodiff/GraphNode.h"
#include "core/ds/String.h"

namespace hahaha
{

using core::ds::Vector;
using core::ds::String;
using ml::TensorPtr;

template<typename T>
class TensorVar : public ad::GraphNode<T>
{
public:
    TensorVar(String name = "") : name_(std::move(name)), ptr_(nullptr) {}

    TensorVar(const std::initializer_list<sizeT> shape, String name = "")
        : name_(std::move(name)), ptr_(new TensorPtr<T>(shape))
    {
        ptr_->refByVar();
    }

    explicit TensorVar(const Vector<sizeT>& shape, String name = "")
        : name_(std::move(name)), ptr_(new TensorPtr<T>(shape))
    {
        ptr_->refByVar();
    }
    explicit TensorVar(const Vector<sizeT>& shape, const T* data, String name = "")
        : name_(std::move(name)), ptr_(new TensorPtr<T>(shape, data))
    {
        ptr_->refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(T scalar, String name = "")
        : name_(std::move(name)), ptr_(new TensorPtr<T>(scalar))
    {
        ptr_->refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(const std::initializer_list<sizeT> shape,
                    std::initializer_list<T> data, String name = "")
        : name_(std::move(name)), ptr_(new TensorPtr<T>(shape, data))
    {
        ptr_->refByVar();
    }

    ~TensorVar() {
        // Here we should handle the custom ref counting logic
        // For now, let's just delete to avoid leaks in the test
        delete ptr_;
    }

private:
    TensorPtr<T>* ptr_;
    String name_;
};

}

#endif // HAHAHA_TENSORVAR_H
