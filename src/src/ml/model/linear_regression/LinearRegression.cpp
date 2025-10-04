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

#include <ml/model/linear_regression/LinearRegression.h>

hhh

namespace hahaha {

    using namespace hahaha::common;

    f32 LinearRegression::predict(const ml::Tensor<f32>& features) const {
        f32 result = _bias;
        result += _weights.dot(features);

        return result;
    }
    bool LinearRegression::save(const Str& filepath) const {
        return true;
    }
    bool LinearRegression::load(const Str& path) {
        return true;
    }
    sizeT LinearRegression::parameterCount() const {
        return _weights.size() + 1;
    }

    Str LinearRegression::modelName() const {
        return Str("LinearRegression");
    }

    ml::Tensor<f32> operator*(f32 lhs, const ml::Tensor<f32>& rhs) {
        auto result = ml::Tensor<f32>(rhs.shape());
        result.copy(rhs);
        for (sizeT i = 0; i < rhs.size(); ++i) {
            result.rawData()[i] *= lhs;
        }
        return result;
    }
    Res<void, BaseErr> LinearRegression::checkStatus(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels)
        const {
        SetRetT(void, BaseErr)

        if (_weights.empty()) {
            Err("Weights are not initialized")
        }

        if (features.shape()[0] != labels.shape()[0]) {
            Err("Mismatch in number of samples between features and labels")
        }

        // if (features.shape())

        for (int i = 1; i < features.shape().size(); i++) {
            if (features.shape()[i] != this->_weights.shape()[i]) {
                Err("invalid shape of features and weights")
            }
        }

        Ok()
    }
    Res<ml::TrainStatistics, BaseErr> LinearRegression::train(
        const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels) {
        SetRetT(ml::TrainStatistics, ml::BaseErr)
        // Initialize weights if not already done
        if (_weights.empty()) {
            _weights = ml::Tensor<f32>(features.shape().subVec(1, features.shape().size() - 1));
        }
        _weights.fill (0);
        _bias = 0.0f;

        if (auto res = checkStatus(features, labels); res.isErr()) {
            Err(res.unwrapErr().message());
        }

        const auto sampleSize = features.shape()[0];

        ml::EmptyTrainStatistics stats;
        // Simple gradient descent
        const auto numEpochs = _epochs;
        const auto learningRate = _learningRate;

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            for (sizeT i = 0; i < sampleSize; ++i) {
                // Forward pass
                ml::Tensor<f32> feature = features.at({i}).unwrap();
                // unwarp will return a Tensor with zero dimension
                const f32 label = labels.at({i}).unwrap().first();
                const f32 prediction                           = predict(feature);
                const f32 error                                = prediction - label;

                // Update weights
                _weights -= learningRate * error * feature;

                // Update bias
                _bias -= learningRate * error;
            }
        }
        Ok(stats)
    }

} // namespace hahaha
