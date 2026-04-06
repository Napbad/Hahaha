//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/26/26.
//
// CPU compute dispatch: operator -> shape/broadcast setup -> per-element Scalar kernels.
// To add Mul/Div/Relu/…: (1) add a small elementwise function, (2) add a case in
// dispatchOnCPU, (3) unary ops can reuse forEachMultiIndex with arity 2 (in + out).
//

#include <optional>
#include <string>
#include <vector>

#include "compute/ComputeDispatcher.h"
#include "compute/ComputeNode.h"
#include "defines.h"
#include "Error.h"
#include "math/Index.h"
#include "math/Scalar.h"
#include "math/TensorShape.h"

namespace h3::core::compute {

namespace {

using ElementwiseBinary = std::expected<void, Error> (*)(math::Scalar& out,
                                                         const math::Scalar& lhs,
                                                         const math::Scalar& rhs);

std::expected<void, Error> elementwiseAdd(math::Scalar& out,
                                          const math::Scalar& lhs,
                                          const math::Scalar& rhs) {
    const math::Scalar sum = lhs + rhs;
    if (sum.dtype() != out.dtype()) {
        return std::unexpected(Error(
            "CPU add: Scalar result dtype does not match output element type "
            "(allocate the output tensor with the promoted element type).",
            ErrorCode::InvalidArgument));
    }
    out = sum;
    return {};
}

std::expected<void, Error> elementwiseSub(math::Scalar& out,
                                          const math::Scalar& lhs,
                                          const math::Scalar& rhs) {
    const math::Scalar v = lhs - rhs;
    if (v.dtype() != out.dtype()) {
        return std::unexpected(Error(
            "CPU sub: Scalar result dtype does not match output element type.",
            ErrorCode::InvalidArgument));
    }
    out = v;
    return {};
}

std::expected<void, Error> elementwiseMul(math::Scalar& out,
                                          const math::Scalar& lhs,
                                          const math::Scalar& rhs) {
    const math::Scalar v = lhs * rhs;
    if (v.dtype() != out.dtype()) {
        return std::unexpected(Error(
            "CPU mul: Scalar result dtype does not match output element type.",
            ErrorCode::InvalidArgument));
    }
    out = v;
    return {};
}

std::expected<void, Error> elementwiseDiv(math::Scalar& out,
                                          const math::Scalar& lhs,
                                          const math::Scalar& rhs) {
    const math::Scalar v = lhs / rhs;
    if (v.dtype() != out.dtype()) {
        return std::unexpected(Error(
            "CPU div: Scalar result dtype does not match output element type.",
            ErrorCode::InvalidArgument));
    }
    out = v;
    return {};
}

/// Walk all multi-indices in row-major nested-loop order (last index varies fastest).
template <class Fn>
void forEachMultiIndex(const math::TensorShape& shape, Fn&& fn) {
    const SizeT rank = shape.rank();
    if (rank == 0) {
        fn(math::Index(std::vector<SizeT>{}));
        return;
    }
    std::vector<SizeT> coord(static_cast<std::size_t>(rank), 0);
    for (;;) {
        fn(math::Index(coord));
        int d = static_cast<int>(rank) - 1;
        for (; d >= 0; --d) {
            ++coord[static_cast<std::size_t>(d)];
            if (coord[static_cast<std::size_t>(d)] < shape[static_cast<SizeT>(d)]) {
                break;
            }
            coord[static_cast<std::size_t>(d)] = 0;
        }
        if (d < 0) {
            break;
        }
    }
}

std::expected<void, Error> expectArity(const std::vector<ComputeNode>& nodes,
                                       const SizeT expected,
                                       const Operator op) {
    if (nodes.size() != expected) {
        return std::unexpected(Error(
            std::string(toString(op)) + " expects " + std::to_string(expected)
                + " compute nodes, got " + std::to_string(nodes.size()),
            ErrorCode::InvalidArgument));
    }
    return {};
}

std::expected<void, Error> expectDispatchDtype(const std::shared_ptr<math::TensorInner>& t,
                                               DataType type,
                                               const char* role) {
    if (t->dataType() != type) {
        return std::unexpected(Error(
            std::string("CPU dispatch: ") + role + " tensor dtype does not match dispatch type.",
            ErrorCode::InvalidArgument));
    }
    return {};
}

std::expected<void, Error> expectSameDeviceThree(const std::shared_ptr<math::TensorInner>& a,
                                                 const std::shared_ptr<math::TensorInner>& b,
                                                 const std::shared_ptr<math::TensorInner>& c,
                                                 const backend::Device& dispatchDevice) {
    const backend::Device& da = a->metadataRef().device;
    const backend::Device& db = b->metadataRef().device;
    const backend::Device& dc = c->metadataRef().device;
    if (!(da == db && db == dc)) {
        return std::unexpected(Error(
            "CPU dispatch: lhs, rhs, and out must share the same device.",
            ErrorCode::InvalidArgument));
    }
    if (!(da == dispatchDevice)) {
        return std::unexpected(Error(
            "CPU dispatch: tensor device does not match dispatch device.",
            ErrorCode::InvalidArgument));
    }
    return {};
}

/// Shared path for binary broadcast ops: nodes = { lhs, rhs, out }.
std::expected<void, Error> runBinaryBroadcastOp(Operator op,
                                                 const std::vector<ComputeNode>& nodes,
                                                 DataType type,
                                                 const backend::Device& device,
                                                 ElementwiseBinary kernel) {
    if (const auto e = expectArity(nodes, 3, op); !e) {
        return e;
    }

    const auto lhsT = nodes[0].tensorInner();
    const auto rhsT = nodes[1].tensorInner();
    const auto outT = nodes[2].tensorInner();

    if (const auto e = expectDispatchDtype(lhsT, type, "lhs"); !e) {
        return e;
    }
    if (const auto e = expectDispatchDtype(rhsT, type, "rhs"); !e) {
        return e;
    }
    if (const auto e = expectDispatchDtype(outT, type, "out"); !e) {
        return e;
    }
    if (const auto e = expectSameDeviceThree(lhsT, rhsT, outT, device); !e) {
        return e;
    }

    const auto outShapeExp = lhsT->shapeRef().broadcastWith(rhsT->shapeRef());
    if (!outShapeExp) {
        return std::unexpected(outShapeExp.error());
    }
    const math::TensorShape& outShape = *outShapeExp;

    if (outT->shapeRef() != outShape) {
        return std::unexpected(Error(
            std::string("Output shape mismatch for ") + toString(op) + ": expected "
                + outShape.toString() + ", got " + outT->shapeRef().toString(),
            ErrorCode::InvalidArgument));
    }

    const ComputeNode lhsView = nodes[0].broadcastView(outShape);
    const ComputeNode rhsView = nodes[1].broadcastView(outShape);

    std::optional<Error> fail;
    forEachMultiIndex(outShape, [&](const math::Index& idx) {
        if (fail.has_value()) {
            return;
        }
        math::Scalar outEl = (*outT)(idx);
        const math::Scalar lhsEl = (*lhsView.tensorInner())(idx);
        const math::Scalar rhsEl = (*rhsView.tensorInner())(idx);
        if (std::expected<void, Error> r = kernel(outEl, lhsEl, rhsEl); !r) {
            fail = std::move(r.error());
        }
    });

    if (fail.has_value()) {
        return std::unexpected(std::move(*fail));
    }
    return {};
}

std::expected<void, Error> dispatchAddOnCPU(const std::vector<ComputeNode>& nodes,
                                            DataType type,
                                            const backend::Device& device) {
    return runBinaryBroadcastOp(Operator::Add, nodes, type, device, elementwiseAdd);
}

std::expected<void, Error> dispatchSubOnCPU(const std::vector<ComputeNode>& nodes,
                                            DataType type,
                                            const backend::Device& device) {
    return runBinaryBroadcastOp(Operator::Sub, nodes, type, device, elementwiseSub);
}

std::expected<void, Error> dispatchMulOnCPU(const std::vector<ComputeNode>& nodes,
                                            DataType type,
                                            const backend::Device& device) {
    return runBinaryBroadcastOp(Operator::Mul, nodes, type, device, elementwiseMul);
}

std::expected<void, Error> dispatchDivOnCPU(const std::vector<ComputeNode>& nodes,
                                            DataType type,
                                            const backend::Device& device) {
    return runBinaryBroadcastOp(Operator::Div, nodes, type, device, elementwiseDiv);
}

} // namespace

std::expected<void, Error> ComputeDispatcher::dispatchOnCPU(const Operator op,
                                                              const std::vector<ComputeNode>& nodes,
                                                              const DataType type,
                                                              const backend::Device device) {

    switch (op) {
    case Operator::Add:
        return dispatchAddOnCPU(nodes, type, device);
    case Operator::Sub:
        return dispatchSubOnCPU(nodes, type, device);
    case Operator::Mul:
        return dispatchMulOnCPU(nodes, type, device);
    case Operator::Div:
        return dispatchDivOnCPU(nodes, type, device);
    default:
        return std::unexpected(Error(
            std::string("Operator ") + toString(op) + " is not supported on CPU",
            ErrorCode::RuntimeError));
    }
}

} // namespace h3::core::compute
