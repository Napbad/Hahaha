//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
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

#ifndef HAHAHA_DISPATCHER_H
#define HAHAHA_DISPATCHER_H

#include <expected>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/Device.h"
#include "compute/ComputeContextV2.h"
#include "compute/DispatchKey.h"
#include "defines.h"
#include "Error.h"
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {

/// Kernel function signatures
using UnaryKernelFunc = void(*)(const math::TensorInner& src, 
                                 math::TensorInner& dst, 
                                 ComputeContext& ctx);
using BinaryKernelFunc = void(*)(const math::TensorInner& src0, 
                                 const math::TensorInner& src1, 
                                 math::TensorInner& dst, 
                                 ComputeContext& ctx);
using TernaryKernelFunc = void(*)(const math::TensorInner& src0, 
                                  const math::TensorInner& src1, 
                                  const math::TensorInner& src2,
                                  math::TensorInner& dst, 
                                  ComputeContext& ctx);

/// Dispatch result containing the resolved kernel function
struct DispatchResult {
    enum class Kind { Unary, Binary, Ternary, None } kind;
    union {
        UnaryKernelFunc unary;
        BinaryKernelFunc binary;
        TernaryKernelFunc ternary;
        void* raw;
    } func;

    DispatchResult() : kind(Kind::None), func{.raw = nullptr} {}

    static DispatchResult makeUnary(UnaryKernelFunc f) {
        DispatchResult r;
        r.kind = Kind::Unary;
        r.func.unary = f;
        return r;
    }

    static DispatchResult makeBinary(BinaryKernelFunc f) {
        DispatchResult r;
        r.kind = Kind::Binary;
        r.func.binary = f;
        return r;
    }

    static DispatchResult makeTernary(TernaryKernelFunc f) {
        DispatchResult r;
        r.kind = Kind::Ternary;
        r.func.ternary = f;
        return r;
    }

    [[nodiscard]] bool isValid() const { return kind != Kind::None; }
    [[nodiscard]] explicit operator bool() const { return isValid(); }
};

/// Global dispatcher singleton that manages kernel registration and dispatch.
/// Uses a flat hash map for O(1) kernel lookup based on DispatchKey.
class Dispatcher {
public:
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    // Prevent copying
    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    /// Register a unary kernel for the given dispatch key
    void registerUnary(DispatchKey key, UnaryKernelFunc func) {
        unaryRegistry_[key] = func;
    }

    /// Register a binary kernel for the given dispatch key
    void registerBinary(DispatchKey key, BinaryKernelFunc func) {
        binaryRegistry_[key] = func;
    }

    /// Register a ternary kernel for the given dispatch key
    void registerTernary(DispatchKey key, TernaryKernelFunc func) {
        ternaryRegistry_[key] = func;
    }

    /// Resolve a kernel for the given dispatch key and arity
    [[nodiscard]] DispatchResult resolve(DispatchKey key, size_t arity) const {
        switch (arity) {
        case 1:
            if (auto it = unaryRegistry_.find(key); it != unaryRegistry_.end()) {
                return DispatchResult::makeUnary(it->second);
            }
            break;
        case 2:
            if (auto it = binaryRegistry_.find(key); it != binaryRegistry_.end()) {
                return DispatchResult::makeBinary(it->second);
            }
            break;
        case 3:
            if (auto it = ternaryRegistry_.find(key); it != ternaryRegistry_.end()) {
                return DispatchResult::makeTernary(it->second);
            }
            break;
        }
        return {};
    }

    /// Dispatch to kernel with runtime checks
    [[nodiscard]] std::expected<void, Error> dispatch(
        Operator op,
        backend::DeviceType device,
        DataType dtype,
        std::vector<utils::OwnPointer<math::TensorInner>>& operands) const;

    /// Check if a kernel is registered
    [[nodiscard]] bool hasKernel(DispatchKey key, size_t arity) const {
        return resolve(key, arity).isValid();
    }

    /// Get registration statistics
    [[nodiscard]] size_t unaryCount() const { return unaryRegistry_.size(); }
    [[nodiscard]] size_t binaryCount() const { return binaryRegistry_.size(); }
    [[nodiscard]] size_t ternaryCount() const { return ternaryRegistry_.size(); }

private:
    Dispatcher() = default;

    std::unordered_map<DispatchKey, UnaryKernelFunc, DispatchKeyHash, DispatchKeyEq> 
        unaryRegistry_;
    std::unordered_map<DispatchKey, BinaryKernelFunc, DispatchKeyHash, DispatchKeyEq> 
        binaryRegistry_;
    std::unordered_map<DispatchKey, TernaryKernelFunc, DispatchKeyHash, DispatchKeyEq> 
        ternaryRegistry_;
};

} // namespace h3::core::compute

#endif // HAHAHA_DISPATCHER_H
