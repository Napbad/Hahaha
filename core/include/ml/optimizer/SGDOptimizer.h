//  Copyright (c) ${original.year} - 2026 Contributors of Hahaha("https://github.com/Napbad/Hahaha")
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_SGDOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
#define HAHAHA_SGDOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
#include "Optimizer.h"

namespace hahaha::ml {
template <typename T> class SGDOptimizer : public Optimizer<T> {
  public:
    /**
     * @brief Construct a new SGDOptimizer.
     * @param parameters List of tensors to optimize.
     * @param learningRate Learning rate.
     */
    SGDOptimizer(std::vector<Tensor<T>> parameters, T learningRate)
        : Optimizer<T>(std::move(parameters), learningRate) {
    }

    /**
     * @brief Performs parameter updates using SGD logic.
     *
     * Iterates through all tracked parameters that require gradients and
     * performs the in-place subtraction of the scaled gradient:
     * theta = theta - lr * gradient
     */
    void step() override {
        T lr = this->getLearningRate();
        for (auto& param : this->getParameters()) {
            if (!param.getRequiresGrad()) {
                continue;
            }

            auto grad = param.grad();
            if (!grad) {
                continue;
            }

            // theta = theta - lr * gradient
            // Use TensorWrapper's axpy for device-neutral in-place update
            param.data()->axpy(-lr, *(grad->data()));
        }
    }
};
} // namespace hahaha::ml_basic_usage

#endif // HAHAHA_SGDOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
