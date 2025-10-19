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
// Created by Napbad on 10/9/25.
//

#ifndef HAHAHA_LINEAR_H
#define HAHAHA_LINEAR_H
#include <nn/layers/Layer.h>

namespace hahaha::ml
{
template <typename T> class Linear final : public Layer<T>
{
  public:
    Linear(sizeT inputSize, sizeT outputSize)
        : weights_(Variable<T>::rand({inputSize, outputSize})),
          bias_(Variable<T>::rand({1, outputSize}))
    {
    }

    Variable<T>* forward(const Variable<T>* input) override
    {
        auto selfShape = weights_.shape();
        auto inputShape = input->shape();
        auto mm = input->matmul(weights_);
        auto mmShape = mm.shape();
        return new Variable<T>(mm + bias_);
    }

    ds::Vector<Variable<T>*> parameters() override
    {
        return {&weights_, &bias_};
    }

  private:
    Variable<T> weights_;
    Variable<T> bias_;
};
} // namespace hahaha::ml

#endif // HAHAHA_LINEAR_H
