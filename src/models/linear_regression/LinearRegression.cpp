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

#include "LinearRegression.h"

namespace hahaha::ml
{

f32 LinearRegression::predict(const Tensor<f32>& features) const
{
    if (weights_.empty())
    {
        return 0;
    }
    f32 result = bias_;
    result += weights_.dot(features);
    std::cout << "features: " << features.rawData() << std::endl;
    std::cout << "result: " << result << std::endl;

    return result;
}
bool LinearRegression::save(const ds::String& filepath) const
{
    return true;
}
bool LinearRegression::load(const ds::String& path)
{
    return true;
}

sizeT LinearRegression::parameterCount() const
{
    return weights_.size() + 1;
}
ds::String LinearRegression::modelName() const
{
    return ds::String("LinearRegression");
}

Res<void, BaseError> LinearRegression::checkStatus(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels)
{
    return Res<void, BaseError>();
}
Res<TrainStatistics, BaseError> LinearRegression::train(const Tensor<f32>& features,
                                                        const Tensor<f32>& labels)
{
    SetRetT(TrainStatistics, BaseError);

    if (weights_.empty())
    {
        weights_ = Tensor<f32>(features.shape().subVector(1, features.shape().size() - 1));
    }
    weights_.fill(1);
    bias_ = 0.0f;

    TrainStatistics stats;
    const auto numEpochs = epochs_;
    const auto learningRate = learningRate_;
    for (sizeT epoch = 0; epoch < numEpochs; ++epoch)
    {
        f32 totalLoss = 0.0f;
        for (sizeT i = 0; i < features.shape()[0]; ++i)
        {
            auto feature_vec = features.data().subVector(i * features.shape()[1], features.shape()[1]);
            Tensor feature(features.shape().subVector(1, features.shape().size() - 1),
                                feature_vec.begin());
            auto label_vec = labels.data().subVector(i, 1);
            const auto label = label_vec[0];

            const auto y_pred = predict(feature);
            const f32 error = y_pred - label;


            weights_ -= feature * (learningRate * error);
            bias_ -= learningRate * error;

            totalLoss += error * error;
        }
        stats.losses.pushBack(totalLoss / static_cast<f32>(features.shape()[0]));
    }
    Ok(stats);
}
} // namespace hahaha::ml
