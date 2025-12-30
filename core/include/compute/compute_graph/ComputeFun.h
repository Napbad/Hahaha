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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_H

#include <memory>

#include "compute/compute_graph/ComputeNode.h"
#include "compute/compute_graph/Operator.h"
#include "math/TensorWrapper.h"
#include "math/ds/TensorData.h"

namespace hahaha::compute {

/**
 * @brief Element-wise addition operator.
 *
 * Forward: z = x + y
 * Backward:
 *   dL/dx = dL/dz * dz/dx = dL/dz * 1
 *   dL/dy = dL/dz * dz/dy = dL/dz * 1
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> add(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->add(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, Operator::Add, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;

    auto gradFun = [lhs, rhs, weakRes]() {
        if (auto res = weakRes.lock()) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                rhs->accumulateGrad(gradPtr);
                rhs->backward();
            }
        }
    };

    resNode->setGradFun(gradFun);
    return resNode;
}

/**
 * @brief Element-wise subtraction operator.
 *
 * Forward: z = x - y
 * Backward:
 *   dL/dx = dL/dz * dz/dx = dL/dz * 1
 *   dL/dy = dL/dz * dz/dy = dL/dz * (-1)
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> sub(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->subtract(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, Operator::Sub, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;

    auto gradFun = [lhs, rhs, weakRes]() {
        if (auto res = weakRes.lock()) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // dL/dy = dL/dz * (-1)
                auto negGrad =
                    std::make_shared<math::TensorWrapper<T>>(-(*gradPtr));
                rhs->accumulateGrad(negGrad);
                rhs->backward();
            }
        }
    };

    resNode->setGradFun(gradFun);
    return resNode;
}

/**
 * @brief Element-wise multiplication operator.
 *
 * Forward: z = x * y
 * Backward:
 *   dL/dx = dL/dz * dz/dx = dL/dz * y
 *   dL/dy = dL/dz * dz/dy = dL/dz * x
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> mul(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->multiply(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, Operator::Mul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;

    auto gradFun = [lhs, rhs, weakRes]() {
        if (auto res = weakRes.lock()) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                // dL/dx = grad_output * y
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*rhs->getData()));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // dL/dy = grad_output * x
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*lhs->getData()));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    };

    resNode->setGradFun(gradFun);
    return resNode;
}

/**
 * @brief Element-wise division operator.
 *
 * Forward: z = x / y
 * Backward:
 *   dL/dx = dL/dz * dz/dx = dL/dz * (1/y)
 *   dL/dy = dL/dz * dz/dy = dL/dz * (-x / y^2)
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> div(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->divide(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;

    auto gradFun = [lhs, rhs, weakRes]() {
        if (auto res = weakRes.lock()) {
            auto gradPtr = res->getGrad();
            auto rhsData = rhs->getData();

            if (lhs->getRequiresGrad()) {
                // dL/dx = grad_output / y
                auto grad = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->divide(*rhsData));
                lhs->accumulateGrad(grad);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // dL/dy = grad_output * (-x / y^2)
                auto lhsData = lhs->getData();
                auto y_sq = rhsData->multiply(*rhsData);
                auto neg_x = -(*lhsData);
                auto local_grad = neg_x.divide(y_sq);
                auto grad = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(local_grad));
                rhs->accumulateGrad(grad);
                rhs->backward();
            }
        }
    };

    resNode->setGradFun(gradFun);
    return resNode;
}

/**
 * @brief Matrix multiplication operator.
 *
 * Forward: Z = X @ Y (Matrix dot product)
 * Backward:
 *   dL/dX = dL/dZ @ Y^T
 *   dL/dY = X^T @ dL/dZ
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> matmul(const std::shared_ptr<ComputeNode<T>>& lhs,
                                       const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->matmul(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, Operator::MatMul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;

    auto gradFun = [lhs, rhs, weakRes]() {
        if (auto res = weakRes.lock()) {
            auto gradPtr = res->getGrad();

            if (lhs->getRequiresGrad()) {
                // dL/dX = dL/dZ @ Y^T
                auto rhsT = rhs->getData()->transpose();
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->matmul(rhsT));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // dL/dY = X^T @ dL/dZ
                auto lhsT = lhs->getData()->transpose();
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    lhsT.matmul(*gradPtr));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    };

    resNode->setGradFun(gradFun);
    return resNode;
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_H
