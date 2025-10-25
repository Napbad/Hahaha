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

#ifndef HIAHIAHIA_SAMPLE_H
#define HIAHIAHIA_SAMPLE_H

#include "core/TensorData.h"

namespace hahaha::ml
{

/**
 * Sample class
 *
 * Represents a single sample in a dataset, containing features and labels
 */
template <typename T> class Sample
{
  public:
    /**
     * Create a new sample
     */
    explicit Sample(TensorData<T> features, TensorData<T> labels)
        : _features(std::move(features)), _labels(std::move(labels))
    {
    }

    /**
     * Get the features
     */
    [[nodiscard]] const TensorData<T>& features() const
    {
        return _features;
    }

    /**
     * Get the labels
     */
    [[nodiscard]] const TensorData<T>& labels() const
    {
        return _labels;
    }

    /**
     * Get mutable features
     */
    TensorData<T>& features()
    {
        return _features;
    }

    /**
     * Get mutable labels
     */
    TensorData<T>& labels()
    {
        return _labels;
    }

    /**
     * Get feature dimension
     */
    [[nodiscard]] size_t featureDim() const
    {
        return _features.size();
    }

    /**
     * Get label dimension
     */
    [[nodiscard]] size_t labelDim() const
    {
        return _labels.size();
    }

    /**
     * Apply a transform function to features
     */
    template <typename F> Sample& transformFeatures(F&& f)
    {
        _features = f(std::move(_features));
        return *this;
    }

    /**
     * Apply a transform function to labels
     */
    template <typename F> Sample& transformLabels(F&& f)
    {
        _labels = f(std::move(_labels));
        return *this;
    }

  private:
    TensorData<T> _features;
    TensorData<T> _labels;
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_SAMPLE_H
