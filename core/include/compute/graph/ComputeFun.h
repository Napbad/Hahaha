// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "common/Operator.h"
#include "compute/graph/ComputeNode.h"
#include "math/TensorWrapper.h"
#include "math/ds/TensorData.h"

namespace hahaha::compute {

/**
 * @brief Helper to create a constant scalar node on the same device as a
 * reference node.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
createScalarNode(const T& value,
                 const std::shared_ptr<ComputeNode<T>>& refNode) {
    auto scalarWrapper = std::make_shared<math::TensorWrapper<T>>(
        math::TensorShape({}), value, refNode->getData()->getDevice());
    return std::make_shared<ComputeNode<T>>(scalarWrapper);
}

// --- Addition ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
add(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->add(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Add, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
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
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> add(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    auto rhs = createScalarNode(rhsScalar, lhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->add(rhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Add, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    gradPtr->sum(),
                    rhs->getData()->getDevice());
                rhs->accumulateGrad(scalarGrad);
                rhs->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
add(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return add(rhs, lhsScalar); // Commutative
}

// --- Subtraction ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
sub(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->subtract(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Sub, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto negGrad =
                    std::make_shared<math::TensorWrapper<T>>(-(*gradPtr));
                rhs->accumulateGrad(negGrad);
                rhs->backward();
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> sub(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    auto rhs = createScalarNode(rhsScalar, lhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->subtract(rhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Sub, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    -(gradPtr->sum()),
                    rhs->getData()->getDevice());
                rhs->accumulateGrad(scalarGrad);
                rhs->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
sub(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto lhs = createScalarNode(lhsScalar, rhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        rhs->getData()->subtractFrom(lhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Sub, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    gradPtr->sum(),
                    lhs->getData()->getDevice());
                lhs->accumulateGrad(scalarGrad);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto negGrad =
                    std::make_shared<math::TensorWrapper<T>>(-(*gradPtr));
                rhs->accumulateGrad(negGrad);
                rhs->backward();
            }
        }
    });
    return resNode;
}

// --- Multiplication ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
mul(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->multiply(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Mul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*rhs->getData()));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*lhs->getData()));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> mul(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    auto rhs = createScalarNode(rhsScalar, lhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->multiply(rhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Mul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes, rhsScalar]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(rhsScalar));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto scalarGradVal = gradPtr->multiply(*lhs->getData()).sum();
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    scalarGradVal,
                    rhs->getData()->getDevice());
                rhs->accumulateGrad(scalarGrad);
                rhs->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
mul(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return mul(rhs, lhsScalar); // Commutative
}

// --- Division ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->divide(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            auto rhsData = rhs->getData();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->divide(*rhsData));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto lhsData = lhs->getData();
                auto rhsDataSquare = rhsData->multiply(*rhsData);
                auto negLhsData = -(*lhsData);
                auto localGrad = negLhsData.divide(rhsDataSquare);
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(localGrad));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> div(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    auto rhs = createScalarNode(rhsScalar, lhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->divide(rhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes, rhsScalar]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->divide(rhsScalar));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // d(x/s)/ds = -x/s^2
                auto rhsSquareData = rhsScalar * rhsScalar;
                auto negLhsData = -(*(lhs->getData()));
                auto localGrad = negLhsData.divide(rhsSquareData);
                auto scalarGradVal = gradPtr->multiply(localGrad).sum();
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    scalarGradVal,
                    rhs->getData()->getDevice());
                rhs->accumulateGrad(scalarGrad);
                rhs->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto lhs = createScalarNode(lhsScalar, rhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        rhs->getData()->divideInto(lhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes, lhsScalar]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            auto rhsData = rhs->getData();
            if (lhs->getRequiresGrad()) {
                // d(s/x)/ds = 1/x
                auto localGrad = std::make_shared<math::TensorWrapper<T>>(
                                      math::TensorShape(rhsData->getShape()),
                                      T(1),
                                      rhsData->getDevice())
                                      ->divide(*rhsData);
                auto scalarGradVal = gradPtr->multiply(localGrad).sum();
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    scalarGradVal,
                    lhs->getData()->getDevice());
                lhs->accumulateGrad(scalarGrad);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // d(s/x)/dx = -s/x^2
                auto rhsSquareData = rhsData->multiply(*rhsData);
                auto negLhsData = -lhsScalar;
                auto localGrad = std::make_shared<math::TensorWrapper<T>>(
                                      math::TensorShape(rhsData->getShape()),
                                      negLhsData,
                                      rhsData->getDevice())
                                      ->divide(rhsSquareData);
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(localGrad));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    });
    return resNode;
}

// --- Matrix Multiplication ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
matmul(const std::shared_ptr<ComputeNode<T>>& lhs,
       const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->matmul(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::MatMul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto rhsT = rhs->getData()->transpose();
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->matmul(rhsT));
                lhs->accumulateGrad(gradLhs);
                lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto lhsT = lhs->getData()->transpose();
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    lhsT.matmul(*gradPtr));
                rhs->accumulateGrad(gradRhs);
                rhs->backward();
            }
        }
    });

    return resNode;
}

// --- Unary Operations ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
reshape(const std::shared_ptr<ComputeNode<T>>& parent,
        const std::vector<size_t>& newShape) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        parent->getData()->reshape(newShape));

    std::shared_ptr<ComputeNode<T>> resNode =
        ComputeNode<T>::createUnary(parent, resData, common::Operator::Reshape);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;
    auto originalShape = parent->getData()->getShape();

    resNode->setGradFun([weakParent, weakRes, originalShape]() {
        auto res = weakRes.lock();
        auto p = weakParent.lock();
        if (res && p) {
            auto gradPtr = res->getGrad();
            if (p->getRequiresGrad()) {
                auto reshapedGrad = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->reshape(originalShape));
                p->accumulateGrad(reshapedGrad);
                p->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
transpose(const std::shared_ptr<ComputeNode<T>>& parent) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        parent->getData()->transpose());

    std::shared_ptr<ComputeNode<T>> resNode = ComputeNode<T>::createUnary(
        parent, resData, common::Operator::Transpose);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;

    resNode->setGradFun([weakParent, weakRes]() {
        auto res = weakRes.lock();
        auto parent = weakParent.lock();
        if (res && parent) {
            auto gradPtr = res->getGrad();
            if (parent->getRequiresGrad()) {
                auto transposedGrad = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->transpose());
                parent->accumulateGrad(transposedGrad);
                parent->backward();
            }
        }
    });
    return resNode;
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_H
