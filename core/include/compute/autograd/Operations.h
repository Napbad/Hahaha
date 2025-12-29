// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0 
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_COMPUTE_AUTOGRAD_OPERATIONS_H
#define HAHAHA_COMPUTE_AUTOGRAD_OPERATIONS_H

#include <memory>
#include <vector>

#include "compute/autograd/Function.h"
#include "compute/autograd/Variable.h"
#include "math/Tensor.h"

namespace hahaha::autograd
{

/**
 * @brief Addition operation for Variables.
 */
template <typename T> class Add : public Function<T>
{
  public:
    /**
     * @brief Apply addition to two variables.
     * @param a First input variable.
     * @param b Second input variable.
     * @return Resulting variable.
     */
    static std::shared_ptr<Variable<T>> apply(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b);

    /**
     * @brief Backward pass for addition.
     * @param grad_output Gradient of the loss with respect to the output.
     * @return Gradients with respect to the two inputs.
     */
    std::vector<std::shared_ptr<hahaha::math::Tensor<T>>> backward(
        std::shared_ptr<hahaha::math::Tensor<T>> grad_output) override;
};

/**
 * @brief Multiplication operation for Variables.
 */
template <typename T> class Mul : public Function<T>
{
  public:
    /**
     * @brief Apply multiplication to two variables.
     * @param a First input variable.
     * @param b Second input variable.
     * @return Resulting variable.
     */
    static std::shared_ptr<Variable<T>> apply(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b);

    /**
     * @brief Backward pass for multiplication.
     * @param grad_output Gradient of the loss with respect to the output.
     * @return Gradients with respect to the two inputs.
     */
    std::vector<std::shared_ptr<hahaha::math::Tensor<T>>> backward(
        std::shared_ptr<hahaha::math::Tensor<T>> grad_output) override;

  private:
    std::shared_ptr<hahaha::math::Tensor<T>> a_data_;
    std::shared_ptr<hahaha::math::Tensor<T>> b_data_;
};

/**
 * @brief Operator overload for variable addition.
 */
template <typename T> std::shared_ptr<Variable<T>> operator+(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b);

/**
 * @brief Operator overload for variable multiplication.
 */
template <typename T> std::shared_ptr<Variable<T>> operator*(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b);

} // namespace hahaha::autograd

#endif // HAHAHA_COMPUTE_AUTOGRAD_OPERATIONS_H
