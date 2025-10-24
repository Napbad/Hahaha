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

#ifndef HIAHIAHIA_DATASET_H
#define HIAHIAHIA_DATASET_H

#include <Error.h>
#include <Res.h>
#include <ds/String.h>
#include <ds/Vector.h>

#include "Sample.h"
#include "core/Tensor.h"

namespace hahaha::ml
{

/**
 * Dataset error class
 */
class DatasetError final : public Error
{
  public:
    explicit DatasetError(ds::String message,
                          ds::String location = ds::String("dataset"))
        : message_(std::move(message)), location_(std::move(location))
    {
    }

    [[nodiscard]] ds::String typeName() const override
    {
        return ds::String("DatasetError");
    }
    [[nodiscard]] ds::String message() const override
    {
        return message_;
    }
    [[nodiscard]] ds::String location() const override
    {
        return location_;
    }
    [[nodiscard]] ds::String toString() const override
    {
        return typeName() + ds::String(": ") + message() + ds::String(" at ")
            + location();
    }

  private:
    ds::String message_;
    ds::String location_;
};

/**
 * Dataset interface
 *
 * Base class for all dataset implementations
 */
template <typename T> class Dataset
{
  public:
    virtual ~Dataset() = default;

    /**
     * Load the dataset
     */
    virtual Res<void, DatasetError> load() = 0;

    /**
     * Get the size of the dataset
     */
    [[nodiscard]] virtual size_t size() const = 0;

    /**
     * Get a sample by index
     */
    [[nodiscard]] virtual Res<Sample<T>, DatasetError>
    get(size_t index) const = 0;

    /**
     * Get the feature dimension
     */
    [[nodiscard]] virtual size_t featureDim() const = 0;

    /**
     * Get the label dimension
     */
    [[nodiscard]] virtual size_t labelDim() const = 0;

    /**
     * Get feature names if available
     */
    [[nodiscard]] virtual ds::Vector<ds::String> featureNames() const
    {
        return {};
    }

    /**
     * Get label names if available
     */
    [[nodiscard]] virtual ds::Vector<ds::String> labelNames() const
    {
        return {};
    }

    /**
     * Get dataset description if available
     */
    [[nodiscard]] virtual ds::String description() const
    {
        return {};
    }
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_DATASET_H
