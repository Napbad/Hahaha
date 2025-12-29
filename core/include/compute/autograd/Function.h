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

#ifndef HAHAHA_COMPUTE_AUTOGRAD_FUNCTION_H
#define HAHAHA_COMPUTE_AUTOGRAD_FUNCTION_H

#include <memory>
#include <vector>

#include "math/Tensor.h"

namespace hahaha::autograd
{

template <typename T> class Variable;

/**
 * @brief Base class for differentiable functions.
 *
 * Each subclass of Function must implement forward and backward passes.
 * During the forward pass, it records necessary information for the backward pass.
 *
 * @tparam T The numeric type of the function's data.
 */
template <typename T> class Function
{
  public:
    Function() = default;
    virtual ~Function() = default;

    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;

    Function(Function&&) noexcept = default;
    Function& operator=(Function&&) noexcept = default;

    /**
     * @brief Perform the backward pass.
     * @param grad_output Gradient of the loss with respect to the output of this function.
     * @return Gradients of the loss with respect to the inputs of this function.
     */
    virtual std::vector<std::shared_ptr<hahaha::math::Tensor<T>>> backward(
        std::shared_ptr<hahaha::math::Tensor<T>> grad_output) = 0;
};

} // namespace hahaha::autograd

#endif // HAHAHA_COMPUTE_AUTOGRAD_FUNCTION_H
