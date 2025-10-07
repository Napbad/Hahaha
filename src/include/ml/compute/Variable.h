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
// Created by napbad on 10/6/25.
//

#ifndef HAHAHA_VARIABLE_H
#define HAHAHA_VARIABLE_H
#include "common/defines/h3defs.h"
#include "ml/common/Tensor.h"
#include <functional>

HHHNamespaceImport

    namespace hahaha::ml {
    template<typename  T>
    class Variable : public Tensor<T> {

    public:
        Variable() = default;
        
        explicit Variable(Tensor<T> tensor, bool requires_grad = false)
            : Tensor<T>(tensor), _grad(tensor.shape()), _requiresGrad(requires_grad) {
            _grad.fill(static_cast<T>(0));
        }

        Variable operator+(const Variable& other) const {
            Variable result(static_cast<Tensor<T>>(*this) + static_cast<Tensor<T>>(other), 
                           _requiresGrad || other._requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator-(const Variable& other) const {
            Variable result(static_cast<Tensor<T>>(*this) - static_cast<Tensor<T>>(other), 
                           _requiresGrad || other._requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator*(const Variable& other) const {
            Variable result(static_cast<Tensor<T>>(*this) * static_cast<Tensor<T>>(other), 
                           _requiresGrad || other._requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator/(const Variable& other) const {
            Variable result(static_cast<Tensor<T>>(*this) / static_cast<Tensor<T>>(other), 
                           _requiresGrad || other._requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        Variable matmul(const Variable& other) const {
            // Simple matrix multiplication for 2D tensors
            // TODO: Implement proper matrix multiplication and gradient
            Variable result(*this, _requiresGrad || other._requiresGrad);
            return result;
        }

        Variable relu() const {
            Tensor<T> result_tensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i) {
                result_tensor[i] = (*this)[i] > static_cast<T>(0) ? (*this)[i] : static_cast<T>(0);
            }
            Variable result(result_tensor, _requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        Variable sigmoid() const {
            Tensor<T> result_tensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i) {
                T val = (*this)[i];
                if constexpr (std::is_floating_point_v<T>) {
                    result_tensor[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
                }
            }
            Variable result(result_tensor, _requiresGrad);
            // TODO: Implement gradient computation
            return result;
        }

        void backward(const Tensor<T>& grad = {}) {
            // Simple backward pass (placeholder implementation)
            if (_requiresGrad) {
                if (grad.empty()) {
                    // Initialize gradient to ones
                    _grad.fill(static_cast<T>(1));
                } else {
                    _grad = grad;
                }
                
                // Call backward function if it exists
                if (_backwardFn) {
                    _backwardFn(_grad);
                }
                
                // Propagate to children
                for (auto& child : _children) {
                    if (child) {
                        child->backward(_grad);
                    }
                }
            }
        }

        void zeroGrad() {
            _grad.fill(static_cast<T>(0));
        }

        const Tensor<T>& grad() const {
            return _grad;
        }

        bool requiresGrad() const {
            return _requiresGrad;
        }

    private:
        Tensor<T> _grad;
        bool _requiresGrad = false;
        std::function<void(const Tensor<T>&)>  _backwardFn;
        ds::Vec<std::shared_ptr<Variable>> _children;

        void buildGraph(
            const ds::Vec<std::shared_ptr<Variable>>& children,
            std::function<Tensor<T>(const Tensor<T>&)> forwardFn,
            std::function<void(const Tensor<T>&)> backwardFn
        ) {
            _children = children;
            _backwardFn = backwardFn;
        }
    };
    
} // namespace hahaha::ml

#endif // HAHAHA_VARIABLE_H
