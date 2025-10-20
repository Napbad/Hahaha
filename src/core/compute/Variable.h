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
// Created by napbad on 10/6/25.
//

#ifndef HAHAHA_VARIABLE_H
#define HAHAHA_VARIABLE_H
#include <cmath>
#include <functional>
#include <memory>

#include "core/defines/h3defs.h"
#include "core/ml/Tensor.h"

HHH_NAMESPACE_IMPORT

namespace hahaha::ml
{

template <typename T> struct VariableNode
{
    Tensor<T> data;
    Tensor<T> grad;
    bool requiresGrad = false;
    ds::Vector<std::weak_ptr<VariableNode>> parents;
    std::function<void(const Tensor<T>&)> backwardFn;
};

template <typename T> class Variable
{

  public:
    Variable() = default;

    explicit Variable(const ds::Vector<sizeT>& shape,
                      const bool requiresGrad = true)
    {
        node_ = std::make_shared<VariableNode<T>>();
        node_->data = Tensor<T>(shape);
        node_->grad = Tensor<T>(shape);
        node_->grad.fill(static_cast<T>(0));
        node_->requiresGrad = requiresGrad;
    }

    Variable(const std::initializer_list<sizeT> shape)
        : Variable(ds::Vector(shape))
    {
    }

    explicit Variable(const Tensor<T>& tensor, const bool requiresGrad = true)
    {
        node_ = std::make_shared<VariableNode<T>>();
        node_->data = tensor;
        node_->grad = Tensor<T>(tensor.shape());
        node_->grad.fill(static_cast<T>(0));
        node_->requiresGrad = requiresGrad;
    }

    // default copy/move keep shared ownership
    Variable(const Variable& other) = default;
    Variable& operator=(const Variable& other) = default;
    Variable(Variable&& other) noexcept = default;
    Variable& operator=(Variable&& other) noexcept = default;
    ~Variable() = default;

    // Accessors mirroring previous Tensor API usage
    [[nodiscard]] const ds::Vector<sizeT>& shape() const { return node_->data.shape(); }
    [[nodiscard]] sizeT size() const { return node_->data.size(); }
    T operator[](sizeT idx) const { return node_->data[idx]; }
    Tensor<T>& tensor() const { return node_->data; }
    // Backward compatibility for examples/tests expecting var.data()[i]
    const ds::Vector<T>& data() const { return node_->data.data(); }
    Tensor<T>& grad() const { return node_->grad; }
    [[nodiscard]] bool requiresGrad() const
    {
        return node_ && node_->requiresGrad;
    }

    // Compatibility conversions to underlying Tensor
    explicit operator Tensor<T>&() { return node_->data; }
    explicit operator const Tensor<T>&() const { return node_->data; }

    // Arithmetic operators with lvalue/rvalue management
    Variable operator+(const Variable& rhs) const &
    {
        return buildBinary(rhs, [](const Tensor<T>& a, const Tensor<T>& b) {
            return a + b;
        }, [](const Tensor<T>& up, const Tensor<T>& /*a*/, const Tensor<T>& /*b*/,
               Tensor<T>& gradA, Tensor<T>& gradB) {
            gradA += up;
            gradB += up;
        });
    }

    Variable operator+(Variable&& rhs) const &
    {
        Variable out = *this + rhs;
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator+(const Variable& rhs) &&
    {
        Variable out = *this + rhs;
        out.keepAlive_.pushBack(std::move(node_));
        return out;
    }

    Variable operator+(Variable&& rhs) &&
    {
        Variable out = *this + rhs;
        out.keepAlive_.pushBack(std::move(node_));
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator-(const Variable& rhs) const &
    {
        return buildBinary(rhs, [](const Tensor<T>& a, const Tensor<T>& b) {
            return a - b;
        }, [](const Tensor<T>& up, const Tensor<T>& /*a*/, const Tensor<T>& /*b*/,
               Tensor<T>& gradA, Tensor<T>& gradB) {
            gradA += up;
            Tensor<T> neg = up;
            neg *= static_cast<T>(-1);
            gradB += neg;
        });
    }

    Variable operator-(Variable&& rhs) const &
    {
        Variable out = *this - rhs;
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator-(const Variable& rhs) &&
    {
        Variable out = *this - rhs;
        out.keepAlive_.pushBack(std::move(node_));
        return out;
    }

    Variable operator-(Variable&& rhs) &&
    {
        Variable out = *this - rhs;
        out.keepAlive_.pushBack(std::move(node_));
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator*(const Variable& rhs) const &
    {
        return buildBinary(rhs, [](const Tensor<T>& a, const Tensor<T>& b) {
            return a * b;
        }, [](const Tensor<T>& up, const Tensor<T>& a, const Tensor<T>& b,
               Tensor<T>& gradA, Tensor<T>& gradB) {
            gradA += up * b;
            gradB += up * a;
        });
    }

    Variable operator*(Variable&& rhs) const &
    {
        Variable out = *this * rhs;
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator*(const Variable& rhs) &&
    {
        Variable out = *this * rhs;
        out.keepAlive_.pushBack(std::move(node_));
        return out;
    }

    Variable operator*(Variable&& rhs) &&
    {
        Variable out = *this * rhs;
        out.keepAlive_.pushBack(std::move(node_));
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator/(const Variable& rhs) const &
    {
        return buildBinary(rhs, [](const Tensor<T>& a, const Tensor<T>& b) {
            return a / b;
        }, [](const Tensor<T>& up, const Tensor<T>& a, const Tensor<T>& b,
               Tensor<T>& gradA, Tensor<T>& gradB) {
            gradA += up / b;
            Tensor<T> b2 = b * b;
            Tensor<T> t = up * a / b2;
            t *= static_cast<T>(-1);
            gradB += t;
        });
    }

    Variable operator/(Variable&& rhs) const &
    {
        Variable out = *this / rhs;
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable operator/(const Variable& rhs) &&
    {
        Variable out = *this / rhs;
        out.keepAlive_.pushBack(std::move(node_));
        return out;
    }

    Variable operator/(Variable&& rhs) &&
    {
        Variable out = *this / rhs;
        out.keepAlive_.pushBack(std::move(node_));
        out.keepAlive_.pushBack(std::move(rhs.node_));
        return out;
    }

    Variable matmul(const Variable& rhs) const &
    {
        Variable out;
        out.node_ = std::make_shared<VariableNode<T>>();
        out.node_->data =
            static_cast<const Tensor<T>&>(node_->data).matmul(rhs.node_->data);
        out.node_->grad = Tensor<T>(out.node_->data.shape());
        out.node_->grad.fill(static_cast<T>(0));
        out.node_->requiresGrad = requiresGrad() || rhs.requiresGrad();
        if (out.node_->requiresGrad)
        {
            out.node_->parents.pushBack(node_);
            out.node_->parents.pushBack(rhs.node_);

            std::weak_ptr<VariableNode<T>> lhsW = node_, rhsW = rhs.node_;
            out.node_->backwardFn = [lhsW, rhsW](const Tensor<T>& upstream) {
                auto lhs = lhsW.lock();
                auto rhsN = rhsW.lock();
                if (lhs && lhs->requiresGrad && rhsN)
                {
                    lhs->grad += upstream.matmul(rhsN->data.transpose());
                }
                if (rhsN && rhsN->requiresGrad && lhs)
                {
                    rhsN->grad += lhs->data.transpose().matmul(upstream);
                }
            };
        }
        return out;
    }

    // Activations on data
    Variable relu() const
    {
        Tensor<T> resultTensor(node_->data.shape());
        for (sizeT i = 0; i < node_->data.size(); ++i)
            resultTensor[i] = node_->data[i] > static_cast<T>(0)
                ? node_->data[i]
                : static_cast<T>(0);

        Variable out(resultTensor, requiresGrad());
        if (out.requiresGrad())
        {
            out.node_->parents.pushBack(node_);
            std::weak_ptr<VariableNode<T>> inpW = node_;
            auto saved = resultTensor;
            out.node_->backwardFn = [inpW, saved](const Tensor<T>& upstream) {
                if (auto inp = inpW.lock(); inp && inp->requiresGrad)
                {
                Tensor<T> mask(upstream.shape());
                for (sizeT i = 0; i < upstream.size(); ++i)
                        mask[i] = saved[i] > static_cast<T>(0)
                        ? static_cast<T>(1)
                        : static_cast<T>(0);
                    inp->grad += upstream * mask;
                }
            };
        }
        return out;
    }

    Variable sigmoid() const
    {
        Tensor<T> resultTensor(node_->data.shape());
        for (sizeT i = 0; i < node_->data.size(); ++i)
        {
            T val = node_->data[i];
            if constexpr (std::is_floating_point_v<T>)
                resultTensor[i] =
                    static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
        }
        Variable out(resultTensor, requiresGrad());
        if (out.requiresGrad())
        {
            out.node_->parents.pushBack(node_);
            std::weak_ptr<VariableNode<T>> inpW = node_;
            auto saved = resultTensor;
            out.node_->backwardFn = [inpW, saved](const Tensor<T>& upstream) {
                if (auto inp = inpW.lock(); inp && inp->requiresGrad)
                {
                    Tensor<T> one(saved.shape());
                one.fill(static_cast<T>(1));
                    Tensor<T> gradInput = upstream * (saved * (one - saved));
                    inp->grad += gradInput;
                }
            };
        }
        return out;
    }

    void zeroGrad()
    {
        if (node_)
            node_->grad.fill(static_cast<T>(0));
    }

    void backward(const Tensor<T>& grad = {})
    {
        if (!node_ || !node_->requiresGrad)
            return;

        Tensor<T> upstream;
        if (grad.empty())
        {
            upstream = Tensor<T>(node_->data.shape());
            upstream.fill(static_cast<T>(1));
        }
        else
        {
            upstream = grad;
        }

        node_->grad += upstream;
        if (node_->backwardFn)
            node_->backwardFn(upstream);
    }

    // In-place compound ops for optimization steps (no graph edges)
    Variable& operator+=(const Tensor<T>& rhs)
    {
        node_->data += rhs;
        return *this;
    }

    Variable& operator-=(const Tensor<T>& rhs)
    {
        node_->data -= rhs;
        return *this;
    }

    Variable& operator*=(const Tensor<T>& rhs)
    {
        node_->data *= rhs;
        return *this;
    }

    Variable& operator/=(const Tensor<T>& rhs)
    {
        node_->data /= rhs;
        return *this;
    }

    Variable& operator+=(const T scalar)
    {
        node_->data = node_->data + scalar;
        return *this;
    }

    Variable& operator-=(const T scalar)
    {
        node_->data = node_->data - scalar;
        return *this;
    }

    Variable& operator*=(const T scalar)
    {
        node_->data = node_->data * scalar;
        return *this;
    }

    Variable& operator/=(const T scalar)
    {
        node_->data = node_->data / scalar;
        return *this;
    }

    // Reductions
    Variable sum() const
    {
        T s = node_->data.sum();
        Tensor<T> outT(ds::Vector<sizeT>({1}));
        outT[0] = s;
        Variable out(outT, requiresGrad());
        if (out.requiresGrad())
        {
            out.node_->parents.pushBack(node_);
            std::weak_ptr<VariableNode<T>> inpW = node_;
            out.node_->backwardFn = [inpW](const Tensor<T>& upstream) {
                if (auto inp = inpW.lock(); inp && inp->requiresGrad)
                {
                    T scale = static_cast<T>(1);
                    // Best-effort: if upstream is scalar, use it
                    if (upstream.size() == 1)
                        scale = upstream[0];
                    Tensor<T> ones(inp->data.shape());
                    ones.fill(scale);
                    inp->grad += ones;
                }
            };
        }
        return out;
    }

    Variable mean() const
    {
        T s = node_->data.sum();
        T m = s / static_cast<T>(node_->data.size());
        Tensor<T> outT(ds::Vector<sizeT>({1}));
        outT[0] = m;
        Variable out(outT, requiresGrad());
        if (out.requiresGrad())
        {
            out.node_->parents.pushBack(node_);
            std::weak_ptr<VariableNode<T>> inpW = node_;
            sizeT n = node_->data.size();
            out.node_->backwardFn = [inpW, n](const Tensor<T>& upstream) {
                if (auto inp = inpW.lock(); inp && inp->requiresGrad)
                {
                    T scale = static_cast<T>(1) / static_cast<T>(n);
                    if (upstream.size() == 1)
                        scale *= upstream[0];
                    Tensor<T> grad(inp->data.shape());
                    grad.fill(scale);
                    inp->grad += grad;
                }
            };
        }
        return out;
    }

    // Static helpers
    static Variable rand(const ds::Vector<sizeT>& shape,
                         const bool requiresGrad = true)
    {
        Tensor<T> t = Tensor<T>::rand(shape);
        return Variable(t, requiresGrad);
    }

    // Helper: common binary op builder
  private:
    // forward Operator, Backend Operator
    template <typename FwdOp, typename BackOp>
    Variable buildBinary(const Variable& rhs, FwdOp fwd, BackOp back) const
    {
        Variable out;
        out.node_ = std::make_shared<VariableNode<T>>();
        out.node_->data = fwd(node_->data, rhs.node_->data);
        out.node_->grad = Tensor<T>(out.node_->data.shape());
        out.node_->grad.fill(static_cast<T>(0));
        out.node_->requiresGrad = requiresGrad() || rhs.requiresGrad();
        if (out.node_->requiresGrad)
        {
            out.node_->parents.pushBack(node_);
            out.node_->parents.pushBack(rhs.node_);
            std::weak_ptr<VariableNode<T>> lhsW = node_, rhsW = rhs.node_;
            out.node_->backwardFn = [lhsW, rhsW, back](
                                         const Tensor<T>& upstream) {
                auto lhs = lhsW.lock();
                auto rhsV = rhsW.lock();
                if (!lhs && !rhsV)
                    return;
                Tensor<T> dummy; // not used when null
                if (lhs && lhs->requiresGrad && rhsV && rhsV->requiresGrad)
                {
                    back(upstream, lhs->data, rhsV->data, lhs->grad, rhsV->grad);
                }
                else if (lhs && lhs->requiresGrad)
                {
                    Tensor<T> ignore(rhsV ? rhsV->data.shape() : lhs->data.shape());
                    back(upstream, lhs->data, ignore, lhs->grad, ignore);
                }
                else if (rhsV && rhsV->requiresGrad)
                {
                    Tensor<T> ignore(lhs ? lhs->data.shape() : rhsV->data.shape());
                    back(upstream, ignore, rhsV->data, ignore, rhsV->grad);
                }
            };
        }
        return out;
    }

  private:
    std::shared_ptr<VariableNode<T>> node_{};
    ds::Vector<std::shared_ptr<VariableNode<T>>> keepAlive_{};
};

} // namespace hahaha::ml

#endif // HAHAHA_VARIABLE_H
