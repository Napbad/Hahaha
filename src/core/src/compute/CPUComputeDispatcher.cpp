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
#include <cstring>
#include <array>
#include <algorithm>
#include <numeric>

#include "compute/ComputeDispatcher.h"
#include "compute/ComputeNode.h"
#include "defines.h"
#include "Error.h"
#include "math/Index.h"
#include "math/Scalar.h"
#include "math/TensorShape.h"

namespace h3::core::compute {

namespace {

// Forward declarations for binary operations
template<typename T>
struct AddOp {
    T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct SubOp {
    T operator()(const T& a, const T& b) const { return a - b; }
};

template<typename T>
struct MulOp {
    T operator()(const T& a, const T& b) const { return a * b; }
};

template<typename T>
struct DivOp {
    T operator()(const T& a, const T& b) const { return a / b; }
};

/// Represents a collapsed dimension after optimization
struct CollapsedDim {
    SizeT size;
    SizeT lhsStride;
    SizeT rhsStride;
    SizeT outStride;
};

/// Calculates offsets for broadcasting operations with optimizations
struct BroadcastOffsetCalculator {
    std::vector<CollapsedDim> collapsedDims;
    SizeT originalRank;
    
    BroadcastOffsetCalculator(
        const math::TensorShape& lhsShape,
        const math::TensorShape& rhsShape,
        const math::TensorShape& outShape,
        const math::TensorStride& lhsStride,
        const math::TensorStride& rhsStride,
        const math::TensorStride& outStride
    ) : originalRank(outShape.rank()) {
        const auto& lhsSizes = lhsShape.sizesRef();
        const auto& rhsSizes = rhsShape.sizesRef();
        const auto& outSizes = outShape.sizesRef();
        
        // Prepare original dimensions with their strides
        std::vector<CollapsedDim> dims(originalRank);
        for (SizeT i = 0; i < originalRank; ++i) {
            // Calculate which dimension to use for strides (accounting for broadcasting)
            SizeT lhsDimIdx = i + lhsShape.rank() - originalRank;
            SizeT rhsDimIdx = i + rhsShape.rank() - originalRank;
            
            // LHS stride: if broadcasted (size=1), stride is 0
            dims[i].lhsStride = (lhsDimIdx >= 0 && lhsDimIdx < lhsShape.rank() && lhsSizes[lhsDimIdx] > 1) ? 
                                lhsStride[lhsDimIdx] : 0;
            
            // RHS stride: if broadcasted (size=1), stride is 0
            dims[i].rhsStride = (rhsDimIdx >= 0 && rhsDimIdx < rhsShape.rank() && rhsSizes[rhsDimIdx] > 1) ? 
                                rhsStride[rhsDimIdx] : 0;
                                
            dims[i].outStride = (i < outStride.size()) ? outStride[i] : 0;
            dims[i].size = (i < outSizes.size()) ? outSizes[i] : 1;
        }
        
        // Collapse dimensions where possible
        collapsedDims.reserve(originalRank);
        
        if (!dims.empty()) {
            collapsedDims.push_back(dims[0]);
            
            for (SizeT i = 1; i < dims.size(); ++i) {
                auto& lastDim = collapsedDims.back();
                
                // Check if we can collapse current dim with the previous one
                // This is possible if the strides are compatible with contiguous access
                bool canCollapse = 
                    (lastDim.lhsStride == 0 || lastDim.lhsStride == dims[i].outStride) && 
                    (lastDim.rhsStride == 0 || lastDim.rhsStride == dims[i].outStride) && 
                    (lastDim.outStride == dims[i].outStride);
                
                if (canCollapse) {
                    // Collapse by multiplying the size and updating strides
                    lastDim.size *= dims[i].size;
                } else {
                    // Keep as separate dimension
                    collapsedDims.push_back(dims[i]);
                }
            }
        }
    }
    
    SizeT getTotalElements() const {
        return std::accumulate(collapsedDims.begin(), collapsedDims.end(), 
                               static_cast<SizeT>(1),
                               [](SizeT acc, const CollapsedDim& dim) { 
                                   return acc * dim.size; 
                               });
    }
};

/// Template-based binary kernel for contiguous tensors (fast path)
template<typename T, typename Op>
void runBinaryKernelContiguous(
    T* __restrict__ outPtr,
    const T* __restrict__ lhsPtr,
    const T* __restrict__ rhsPtr,
    const SizeT totalElements
) {
    Op op;
    #pragma omp parallel for if(totalElements > 10000)
    for (SizeT i = 0; i < totalElements; ++i) {
        outPtr[i] = op(lhsPtr[i], rhsPtr[i]);
    }
}

/// Template-based binary kernel with optimized stride support (general path)
template<typename T, typename Op>
void runBinaryKernelWithStrides(
    T* __restrict__ outPtr,
    const T* __restrict__ lhsPtr,
    const T* __restrict__ rhsPtr,
    const BroadcastOffsetCalculator& calc
) {
    Op op;
    
    const SizeT collapsedRank = calc.collapsedDims.size();
    
    // Use fixed-size arrays instead of vectors to avoid heap allocations
    std::array<SizeT, 8> coords{};  // Initialize to 0
    std::array<SizeT, 8> outOffsets{};
    std::array<SizeT, 8> lhsOffsets{};
    std::array<SizeT, 8> rhsOffsets{};
    
    // Ensure we don't exceed our max rank
    if (collapsedRank > 8) {
        // Fallback to the less optimized version for very high-rank tensors
        // This shouldn't happen in practice given our max rank assumption
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions");
    }
    
    // Initialize all arrays to 0
    for (SizeT i = 0; i < collapsedRank; ++i) {
        coords[i] = 0;
        outOffsets[i] = 0;
        lhsOffsets[i] = 0;
        rhsOffsets[i] = 0;
    }
    
    SizeT totalElements = calc.getTotalElements();
    
    // Process elements using optimized coordinate walking
    #pragma omp parallel for if(totalElements > 10000)
    for (SizeT elemIdx = 0; elemIdx < totalElements; ++elemIdx) {
        // Calculate current offsets based on coordinates
        SizeT outOffset = 0;
        SizeT lhsOffset = 0;
        SizeT rhsOffset = 0;
        
        for (SizeT i = 0; i < collapsedRank; ++i) {
            const auto& dim = calc.collapsedDims[i];
            outOffset += coords[i] * dim.outStride;
            lhsOffset += coords[i] * dim.lhsStride;
            rhsOffset += coords[i] * dim.rhsStride;
        }
        
        // Perform the operation
        outPtr[outOffset] = op(lhsPtr[lhsOffset], rhsPtr[rhsOffset]);
        
        // Increment coordinates in row-major order (incremental update)
        int d = static_cast<int>(collapsedRank) - 1;
        for (; d >= 0; --d) {
            ++coords[d];
            if (coords[d] < calc.collapsedDims[d].size) {
                break;
            }
            coords[d] = 0;  // Reset coordinate
        }
        
        // If we've processed all elements, break early
        if (d < 0) {
            break;
        }
    }
}

/// Main templated binary kernel dispatcher
template<typename T, typename Op>
std::expected<void, Error> runBinaryKernel(
    std::vector<ComputeNode>& nodes,
    const backend::Device& device,
    Op op
) {
    if (nodes.size() != 3) {
        return std::unexpected(Error(
            "Binary operation expects 3 compute nodes (lhs, rhs, out)",
            ErrorCode::InvalidArgument));
    }

    const auto& lhsT = nodes[0].tensorInner();
    const auto& rhsT = nodes[1].tensorInner();
    auto outT = nodes[2].tensorInner(); // Note: may be null initially
    
    // Calculate broadcasted output shape
    const auto outShapeExp = lhsT->shapeRef().broadcastWith(rhsT->shapeRef());
    if (!outShapeExp) {
        return std::unexpected(outShapeExp.error());
    }
    
    // Create output tensor if it doesn't exist
    if (outT == nullptr) {
        nodes[2].setTensorInner(math::TensorInner(
            outShapeExp.value(),
            math::TensorMetadata{
                .dataType = lhsT->dataType(),
                .device = lhsT->device()
            }));
        outT = nodes[2].tensorInner();
    }
    
    // Validate device compatibility
    if (!(lhsT->device() == rhsT->device() && rhsT->device() == device)) {
        return std::unexpected(Error(
            "CPU dispatch: lhs, rhs, and dispatch device must match.",
            ErrorCode::InvalidArgument));
    }
    
    const math::TensorShape& outShape = outShapeExp.value();
    
    // Check shape compatibility
    if (outT->shapeRef() != outShape) {
        return std::unexpected(Error(
            "Output shape mismatch: expected " + outShape.toString() + 
            ", got " + outT->shapeRef().toString(),
            ErrorCode::InvalidArgument));
    }
    
    // Get raw pointers to data
    T* __restrict__ outData = outT->storageRef().data<T>() + outT->offset();
    const T* __restrict__ lhsData = lhsT->storageRef().data<const T>() + lhsT->offset();
    const T* __restrict__ rhsData = rhsT->storageRef().data<const T>() + rhsT->offset();
    
    // Fast path: if all tensors are contiguous and have the same shape, use simple loop
    if (lhsT->shapeRef() == outShape && 
        rhsT->shapeRef() == outShape &&
        lhsT->strideRef() == math::TensorStride(outShape) &&
        rhsT->strideRef() == math::TensorStride(outShape) &&
        outT->strideRef() == math::TensorStride(outShape)) {

        const SizeT totalElements = outShape.getTotalSize();
        runBinaryKernelContiguous<T, Op>(outData, lhsData, rhsData, totalElements);
    } else {
        // General path: use stride-based calculation for broadcasting
        const auto calculator = BroadcastOffsetCalculator(
            lhsT->shapeRef(),
            rhsT->shapeRef(),
            outShape,
            lhsT->strideRef(),
            rhsT->strideRef(),
            outT->strideRef()
        );
        
        runBinaryKernelWithStrides<T, Op>(
            outData, lhsData, rhsData, calculator
        );
    }
    
    return {};
}

/// Type-erased dispatcher that selects the correct template instantiation
std::expected<void, Error> dispatchTypedBinaryOperation(
    std::vector<ComputeNode>& nodes,
    DataType type,
    const backend::Device& device,
    const Operator op
) {
    switch (op) {
        case Operator::Add:
            switch (type) {
                case DataType::Float32: 
                    return runBinaryKernel<float, AddOp<float>>(nodes, device, AddOp<float>());
                case DataType::Float64:
                    return runBinaryKernel<double, AddOp<double>>(nodes, device, AddOp<double>());
                case DataType::Int32:
                    return runBinaryKernel<Int32, AddOp<Int32>>(nodes, device, AddOp<Int32>());
                case DataType::Int64:
                    return runBinaryKernel<Int64, AddOp<Int64>>(nodes, device, AddOp<Int64>());
                default:
                    break;
            }
            break;
        case Operator::Sub:
            switch (type) {
                case DataType::Float32: 
                    return runBinaryKernel<float, SubOp<float>>(nodes, device, SubOp<float>());
                case DataType::Float64:
                    return runBinaryKernel<double, SubOp<double>>(nodes, device, SubOp<double>());
                case DataType::Int32:
                    return runBinaryKernel<Int32, SubOp<Int32>>(nodes, device, SubOp<Int32>());
                case DataType::Int64:
                    return runBinaryKernel<Int64, SubOp<Int64>>(nodes, device, SubOp<Int64>());
                default:
                    break;
            }
            break;
        case Operator::Mul:
            switch (type) {
                case DataType::Float32: 
                    return runBinaryKernel<float, MulOp<float>>(nodes, device, MulOp<float>());
                case DataType::Float64:
                    return runBinaryKernel<double, MulOp<double>>(nodes, device, MulOp<double>());
                case DataType::Int32:
                    return runBinaryKernel<Int32, MulOp<Int32>>(nodes, device, MulOp<Int32>());
                case DataType::Int64:
                    return runBinaryKernel<Int64, MulOp<Int64>>(nodes, device, MulOp<Int64>());
                default:
                    break;
            }
            break;
        case Operator::Div:
            switch (type) {
                case DataType::Float32: 
                    return runBinaryKernel<float, DivOp<float>>(nodes, device, DivOp<float>());
                case DataType::Float64:
                    return runBinaryKernel<double, DivOp<double>>(nodes, device, DivOp<double>());
                case DataType::Int32:
                    return runBinaryKernel<Int32, DivOp<Int32>>(nodes, device, DivOp<Int32>());
                case DataType::Int64:
                    return runBinaryKernel<Int64, DivOp<Int64>>(nodes, device, DivOp<Int64>());
                default:
                    break;
            }
            break;
        default:
            break;
    }
    
    return std::unexpected(Error(
        std::string("Operator ") + toString(op) + " with data type " + std::to_string(static_cast<int>(type)) + 
        " is not supported on CPU",
        ErrorCode::RuntimeError));
}

} // namespace

std::expected<void, Error> ComputeDispatcher::dispatchOnCPU(const Operator op,
    std::vector<ComputeNode>& nodes,
    const DataType type,
    const backend::Device device) {

    return dispatchTypedBinaryOperation(nodes, type, device, op);
}

} // namespace h3::core::compute