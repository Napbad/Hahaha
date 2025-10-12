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
#include <cmath>
#include <functional>

#include "core/defines/h3defs.h"
#include "core/ml/Tensor.h"

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{
    template <typename T> class Variable : public Tensor<T>
    {

      public:
        Variable() = default;

        explicit Variable(Tensor<T> tensor, const bool requiresGrad = true)
            : Tensor<T>(tensor), grad_(tensor.shape()), requiresGrad_(requiresGrad)
        {
            grad_.fill(static_cast<T>(0));
        }

        explicit Variable(const ds::Vector<sizeT>& shape) : Tensor<T>(shape)
        {
            grad_ = Tensor<T>(shape);
        }

        // Copy constructor
        Variable(const Variable& other)
            : Tensor<T>(other), grad_(other.grad_), requiresGrad_(other.requiresGrad_),
              backwardFn_(other.backwardFn_), children_(other.children_)
        {
        }

        Variable& operator=(const Variable& other) = default;
        Variable(Variable&& other) = default;
        Variable& operator=(Variable&& other) = default;
        ~Variable() = default;

        Variable operator+(Variable& other)
        {
            Variable result(static_cast<Tensor<T>>(*this) + static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            children_.pushBack(std::make_shared<Variable>(result));
            grad_.fill(1);
            other.grad_.fill(1);
            return result;
        }

        Variable operator-(Variable& other)
        {
            Variable result(static_cast<Tensor<T>>(*this) - static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            children_.pushBack(std::make_shared<Variable>(other));
            grad_.fill(1);
            other.grad_.fill(-1);
            return result;
        }

        Variable operator*(Variable& other)
        {
            Variable result(static_cast<Tensor<T>>(*this) * static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            children_.pushBack(std::make_shared<Variable>(other));
            grad_.copy(other);
            other.grad_.copy(this);
            return result;
        }

        Variable operator/(Variable& other)
        {
            Variable result(static_cast<Tensor<T>>(*this) / static_cast<Tensor<T>>(other),
                            requiresGrad_ || other.requiresGrad_);
            children_.pushBack(std::make_shared<Variable>(other));
            grad_.fill(1);
            for (sizeT i = 0; i < other.shape().size(); i++)
                grad_[i] /= other.shape()[i];

            other.grad_.copy(this);
            other.grad_ *= -1;
            other.grad_ = other.grad_ / this / this;

            return result;
        }

        Variable matmul(const Variable& other) const
        {
            Tensor<T> resultTensor = static_cast<const Tensor<T>&>(*this).matmul(other);
            Variable result(resultTensor, requiresGrad_ || other.requiresGrad_);
            // Backward function needs to handle both inputs
            if (result.requiresGrad()) {
                result.backwardFn_ = [this, other_ref = other](const Tensor<T>& grad) mutable {
                    if (this->requiresGrad()) {
                        this->grad() += grad.matmul(other_ref.transpose());
                    }
                    if (other_ref.requiresGrad()) {
                        other_ref.grad() += this->transpose().matmul(grad);
                    }
                };
            }

            return result;
        }

        Variable relu() const
        {
            Tensor<T> resultTensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i)
                resultTensor[i] = (*this)[i] > static_cast<T>(0) ? (*this)[i] : static_cast<T>(0);

            Variable result(resultTensor, requiresGrad_);

            if (requiresGrad_)
                for (sizeT i = 0; i < this->grad().size(); ++i)
                    this->grad()[i] = (*this)[i] > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);

            return result;
        }

        Variable sigmoid() const
        {
            Tensor<T> resultTensor(this->shape());
            for (sizeT i = 0; i < this->size(); ++i)
            {
                T val = (*this)[i];
                if constexpr (std::is_floating_point_v<T>)
                    resultTensor[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
            }
            Variable result(resultTensor, requiresGrad_);
            grad_ = (resultTensor - 1) * resultTensor;
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
                    backwardFn_(grad_);

                // Propagate to children
                for (auto& child : children_)
                    if (child)
                        child->backward(grad_);
            }
        }


        void zeroGrad()
        {
            grad_.fill(static_cast<T>(0));
        }

        Tensor<T>& grad() const
        {
            return grad_;
        }

        [[nodiscard]] bool requiresGrad() const
        {
            return requiresGrad_;
        }

      private:
        mutable Tensor<T> grad_;
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
