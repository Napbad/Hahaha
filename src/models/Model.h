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
// Created by Napbad on 7/14/25.
//

#ifndef HIAHIAHIA_MODEL_H
#define HIAHIAHIA_MODEL_H

#include <../core/Tensor.h>
#include <Error.h>
#include <Res.h>
#include <ds/String.h>
#include <ml/trainStatistics.h>

namespace hahaha::ml
{

using namespace hahaha::core;
class Model
{
  public:
    virtual ~Model() = default;

    // Train the model with given features and labels
    virtual ml::TrainStatistics train(const ml::Tensor<f32>& features,
                                      const ml::Tensor<f32>& labels) = 0;

    // Make a prediction given input features
    [[nodiscard]] virtual f32
    predict(const ml::Tensor<f32>& features) const = 0;

    // Save the model to a file
    [[nodiscard]] virtual bool save(const ds::String& filepath) const = 0;

    // Load the model from a file
    [[nodiscard]] virtual bool load(const ds::String& filepath) = 0;

    // Get the number of parameters in the model
    [[nodiscard]] virtual sizeT parameterCount() const = 0;

    // Get a name describing the model type
    [[nodiscard]] virtual ds::String modelName() const = 0;
};
} // namespace hahaha::ml

#endif // HIAHIAHIA_MODEL_H
