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
// Created by root on 10/25/25.
//

#ifndef HAHAHA_TENSORVAROP_H
#define HAHAHA_TENSORVAROP_H

namespace hahaha
{
enum class TensorVarOpType
{
    // Basic arithmetic operations
    Add,        // Addition (a + b)
    Sub,        // Subtraction (a - b)
    Mul,        // Element-wise multiplication (a * b)
    Div,        // Element-wise division (a / b, floating-point division)
    DivFloor,   // Floor division (round down, a // b)
    DivCeil,    // Ceiling division (round up)
    Mod,        // Modulus (a % b, sign matches divisor)
    Remainder,  // Remainder (a rem b, sign matches dividend)
    Neg,        // Negation (-a)
    Pow,        // Exponentiation (a^b)
    MatMul,     // Matrix multiplication (a @ b)

    // Exponential and logarithmic operations
    Exp,        // Natural exponential (e^a)
    Exp2,       // Base-2 exponential (2^a)
    Pow10,      // Base-10 exponential (10^a)
    Log,        // Natural logarithm (ln(a))
    Log2,       // Base-2 logarithm (log2(a))
    Log10,      // Base-10 logarithm (log10(a))
    Log1p,      // Logarithm of (1 + a) (avoids precision loss near 0)

    // Trigonometric functions
    Sin,        // Sine (sin(a))
    Cos,        // Cosine (cos(a))
    Tan,        // Tangent (tan(a))
    Asin,       // Arcsine (arcsin(a))
    Acos,       // Arccosine (arccos(a))
    Atan,       // Arctangent (arctan(a))
    Atan2,      // Two-argument arctangent (arctan(y, x))

    // Hyperbolic functions
    Sinh,       // Hyperbolic sine (sinh(a))
    Cosh,       // Hyperbolic cosine (cosh(a))
    Tanh,       // Hyperbolic tangent (tanh(a))
    Asinh,      // Inverse hyperbolic sine (arsinh(a))
    Acosh,      // Inverse hyperbolic cosine (arcosh(a))
    Atanh,      // Inverse hyperbolic tangent (artanh(a))

    // Basic functions
    Abs,        // Absolute value (|a|)
    Sqrt,       // Square root (√a)
    Cbrt,       // Cube root (a^(1/3))
    Floor,      // Floor (round down to nearest integer)
    Ceil,       // Ceiling (round up to nearest integer)
    Round,      // Rounding (round to nearest integer)
    Trunc,      // Truncation (remove fractional part)
    Sigmoid,    // Sigmoid function (1 / (1 + exp(-a)))

    // Statistical operations
    Max,        // Element-wise maximum (or along axis)
    Min,        // Element-wise minimum (or along axis)
    Sum,        // Summation (accumulate along axis)
    Prod,       // Product (multiply along axis)
    Mean,       // Mean (average along axis)
    Variance,   // Variance (along axis)
    Std,        // Standard deviation (along axis)
    Cumsum,     // Cumulative sum (along axis)
    Cumprod,    // Cumulative product (along axis)

    // Comparison operations (return boolean tensor)
    Equal,      // Equal (a == b)
    NotEqual,   // Not equal (a != b)
    Greater,    // Greater than (a > b)
    Less,       // Less than (a < b)
    GreaterEqual, // Greater than or equal (a >= b)
    LessEqual,   // Less than or equal (a <= b)

    // Logical operations (for boolean tensors)
    And,        // Logical AND (a && b)
    Or,         // Logical OR (a || b)
    Not,        // Logical NOT (!a)
    Xor,        // Logical XOR (a ^ b)

    None        // No operation
};

} // namespace hahaha

#endif // HAHAHA_TENSORVAROP_H
