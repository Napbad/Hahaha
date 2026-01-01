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

#ifndef HAHAHA_COMMON_OPERATOR_H
#define HAHAHA_COMMON_OPERATOR_H

namespace hahaha::common {

/**
 * @brief Enumeration of all supported operations in the computational graph.
 *
 * This enum is used by ComputeNode to identify which operation produced it,
 * allowing for metadata tracking and potentially different execution paths.
 */
enum class Operator {
    Add = 0,   /**< Element-wise addition. */
    Sub,       /**< Element-wise subtraction. */
    Mul,       /**< Element-wise multiplication. */
    MatMul,    /**< Matrix multiplication (dot product). */
    Div,       /**< Element-wise division. */
    Pow,       /**< Power operation (x^y). */
    Sqrt,      /**< Square root. */
    Exp,       /**< Exponential (e^x). */
    Log,       /**< Natural logarithm (ln). */
    Tanh,      /**< Hyperbolic tangent. */
    Sigmoid,   /**< Sigmoid activation function. */
    Relu,      /**< Rectified Linear Unit. */
    LeakyRelu, /**< Leaky Rectified Linear Unit. */
    Softmax,   /**< Softmax activation. */
    Max,       /**< Maximum value. */
    Min,       /**< Minimum value. */
    Mean,      /**< Mean value calculation. */
    Sum,       /**< Summation across dimensions. */
    Concat,    /**< Concatenation of tensors. */
    Reshape,   /**< Change tensor shape. */
    Flatten,   /**< Flatten tensor to 1D. */
    Transpose, /**< Transpose dimensions. */
    None       /**< No operation (leaf node). */
};

} // namespace hahaha::common
#endif // HAHAHA_COMMON_OPERATOR_H
