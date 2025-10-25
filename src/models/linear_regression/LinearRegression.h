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

#ifndef HIAHIAHIA_LINEARREGRESSION_H
#define HIAHIAHIA_LINEARREGRESSION_H

#include "core/TensorData.h"
#include "models/Model.h"

namespace hahaha::ml
{

class LinearRegression final : public Model
{
  public:
    LinearRegression() = default;

    LinearRegression(const f32 learningRate, const sizeT epochs)
        : learningRate_(learningRate), epochs_(epochs)
    {
    }

    void checkStatus(const TensorData<f32>& features,
                     const TensorData<f32>& labels) const;
    ml::TrainStatistics train(const TensorData<f32>& features,
                              const TensorData<f32>& labels) override;
    [[nodiscard]] f32 predict(const TensorData<f32>& features) const override;
    [[nodiscard]] bool save(const String& filepath) const override;
    bool load(const String& path) override;
    [[nodiscard]] sizeT parameterCount() const override;
    [[nodiscard]] String modelName() const override;

    void setLearningRate(const f32 learningRate)
    {
        learningRate_ = learningRate;
    }
    void setEpochs(const sizeT epochs)
    {
        epochs_ = epochs;
    }
    [[nodiscard]] f32 learningRate() const
    {
        return learningRate_;
    }

    [[nodiscard]] sizeT epochs() const
    {
        return epochs_;
    }

    void setWeights(const TensorData<f32>& weights)
    {
        weights_ = weights;
    }
    [[nodiscard]] const TensorData<f32>& weights() const
    {
        return weights_;
    }
    void setBias(const f32 bias)
    {
        bias_ = bias;
    }
    [[nodiscard]] f32 bias() const
    {
        return bias_;
    }

  private:
    TensorData<f32> weights_;
    f32 bias_{0.0f};

    f32 learningRate_{0.01f};
    sizeT epochs_{1000};
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_LINEARREGRESSION_H
