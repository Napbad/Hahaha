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

namespace hahaha {

    using namespace hahaha::common;

    f32 LinearRegression::predict(const ml::Tensor<f32>& features) const {
        f32 result = _bias;
        result += _weights.dot(features);

        return result;
    }

    ml::Tensor<f32> operator*(f32 lhs, const ml::Tensor<f32>& rhs) {
        auto result = ml::Tensor<f32>(rhs.shape());
        result.copy(rhs);
        for (sizeT i = 0; i < rhs.size(); ++i) {
            result.rawData()[i] *= lhs;
        }
        return result;
    }
    void LinearRegression::train(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels) {
        // Initialize weights if not already done
        _bias = 0.0f;
        _weights.fill(0);

        // Simple gradient descent
        const f32 learningRate = 0.01f;
        const int numEpochs      = 100;

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            for (sizeT i = 0; i < features.size(); ++i) {
                // Forward pass
                Res<ml::Tensor<f32>, IndexOutOfBoundError> res = features.at({1});
                const f32 prediction = predict(res.unwrap());
                const f32 error      = prediction - labels.index({i}).unwrap<f32>();

                // Update weights
                _weights -= learningRate * error * features.at({i}).unwrap();

                // Update bias
                _bias -= learningRate * error;
            }
        }
    }

} // namespace hahaha
