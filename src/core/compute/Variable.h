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
#include <functional>

#include "common/defines/h3defs.h"
#include "ml/common/Tensor.h"

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{
    template <typename T> class Variable : public Tensor<T>
    {

      public:
        Variable() = default;

        explicit Variable(Tensor<T> tensor, bool requiresGrad = false)
            : Tensor<T>(tensor), grad_(tensor.shape()), requiresGrad_(requiresGrad)
        {
            grad_.fill(static_cast<T>(0));
        }

        Variable operator+(const Variable& other) const
        {
            Variable result(static_cast<Tensor<T>>(*this) + static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator-(const Variable& other) const
        {
            Variable result(static_cast<Tensor<T>>(*this) - static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator*(const Variable& other) const
        {
            Variable result(static_cast<Tensor<T>>(*this) * static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        Variable operator/(const Variable& other) const
        {
            Variable result(static_cast<Tensor<T>>(*this) / static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        Variable matmul(const Variable& other) const
        {
            // Simple matrix multiplication for 2D tensors
            // TODO: Implement proper matrix multiplication and gradient
            Variable result(*this, requiresGrad_ || other.requiresGrad_);
            return result;
        }

        Variable relu() const
        {
            Tensor<T> result_tensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i)
            {
                result_tensor[i] = (*this)[i] > static_cast<T>(0) ? (*this)[i] : static_cast<T>(0);
            }
            Variable result(result_tensor, requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        Variable sigmoid() const
        {
            Tensor<T> result_tensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i)
            {
                T val = (*this)[i];
                if constexpr (std::is_floating_point_v<T>)
                {
                    result_tensor[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
                }
            }
            Variable result(result_tensor, requiresGrad_);
            // TODO: Implement gradient computation
            return result;
        }

        void backward(const Tensor<T>& grad = {})
        {
            // Simple backward pass (placeholder implementation)
            if (requiresGrad_)
            {
                if (grad.empty())
                {
                    // Initialize gradient to ones
                    grad_.fill(static_cast<T>(1));
                }
                else
                {
                    grad_ = grad;
                }

                // Call backward function if it exists
                if (backwardFn_)
                {
                    backwardFn_(grad_);
                }

                // Propagate to children
                for (auto& child : children_)
                {
                    if (child)
                    {
                        child->backward(grad_);
                    }
                }
            }
        }

        void zeroGrad()
        {
            grad_.fill(static_cast<T>(0));
        }

        const Tensor<T>& grad() const
        {
            return grad_;
        }

        bool requiresGrad() const
        {
            return requiresGrad_;
        }

      private:
        Tensor<T> grad_;
        bool requiresGrad_ = false;
        std::function<void(const Tensor<T>&)> backwardFn_;
        ds::Vector<std::shared_ptr<Variable>> children_;

        void buildGraph(const ds::Vector<std::shared_ptr<Variable>>& children,
                        std::function<Tensor<T>(const Tensor<T>&)> forwardFn,
                        std::function<void(const Tensor<T>&)> backwardFn)
        {
            children_ = children;
            backwardFn_ = backwardFn;
        }
    };

} // namespace hahaha::ml

#endif // HAHAHA_VARIABLE_H
