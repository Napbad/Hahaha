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

#ifndef HIAHIAHIA_DATALOADER_H
#define HIAHIAHIA_DATALOADER_H

#include <common/Res.h>
#include <common/ds/Vector.h>
#include <common/Error.h>
#include <memory>
#include <ml/util/dataset/Dataset.h>
#include <ml/util/dataset/Sample.h>
#include <random>

namespace hahaha::ml
{

/**
 * DataLoader error class
 */
class DataLoaderError : public Error
{
  public:
    explicit DataLoaderError(ds::String message, ds::String location = ds::String("DataLoader"))
        : message_(std::move(message)), location_(std::move(location))
    {
    }

    [[nodiscard]] ds::String typeName() const override
    {
        return ds::String("DataLoaderError");
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
        return typeName() + ds::String(": ") + message() + ds::String(" at ") + location();
    }

  private:
    ds::String message_;
    ds::String location_;
};

/**
 * DataLoader class
 *
 * Provides functionality to load data in batches with optional shuffling
 * and preprocessing capabilities.
 */
template <typename T> class DataLoader
{
  public:
    /**
     * Create a new DataLoader
     */
    DataLoader(std::shared_ptr<Dataset<T>> dataset, size_t batchSize, bool shuffle = true, bool dropLast = false)
        : dataset_(std::move(dataset)), batchSize_(batchSize), shuffle_(shuffle), dropLast_(dropLast), currentIndex_(0)
    {
        if (shuffle_)
        {
            shuffleIndices();
        }
        else
        {
            indices_.reserve(dataset_->size());
            for (size_t i = 0; i < dataset_->size(); ++i)
            {
                indices_.pushBack(i);
            }
        }
    }

    /**
     * Reset the iterator to the beginning and optionally reshuffle
     */
    void reset()
    {
        currentIndex_ = 0;
        if (shuffle_)
        {
            shuffleIndices();
        }
    }

    /**
     * Get the next batch of samples
     */
    Res<ds::Vector<Sample<T>>, DataLoaderError> nextBatch()
    {
        SetRetT(ds::Vector<Sample<T>>, DataLoaderError);

        if (currentIndex_ >= dataset_->size())
        {
            Err(DataLoaderError(ds::String("No more batches available")));
        }

        ds::Vector<Sample<T>> batch;
        const size_t remaining = std::min(batchSize_, dataset_->size() - currentIndex_);

        // If drop_last is true and this is the last incomplete batch, return error
        if (dropLast_ && remaining < batchSize_)
        {
            Err(DataLoaderError(ds::String("Dropping last incomplete batch")));
        }

        // Load the batch
        for (size_t i = 0; i < remaining; ++i)
        {
            auto sampleRes = dataset_->get(indices_[currentIndex_ + i]);
            if (sampleRes.isErr())
            {
                Err(DataLoaderError(sampleRes.unwrapErr().message()));
            }
            batch.pushBack(sampleRes.unwrap());
        }

        currentIndex_ += remaining;
        Ok(std::move(batch));
    }

    /**
     * Check if there are more batches available
     */
    [[nodiscard]] bool hasNext() const
    {
        if (dropLast_)
        {
            return currentIndex_ + batchSize_ <= dataset_->size();
        }
        return currentIndex_ < dataset_->size();
    }

    /**
     * Get the total number of batches
     */
    [[nodiscard]] size_t numBatches() const
    {
        if (dropLast_)
        {
            return dataset_->size() / batchSize_;
        }
        return (dataset_->size() + batchSize_ - 1) / batchSize_;
    }

    /**
     * Get the batch size
     */
    [[nodiscard]] size_t batchSize() const
    {
        return batchSize_;
    }

    /**
     * Get the dataset size
     */
    [[nodiscard]] size_t datasetSize() const
    {
        return dataset_->size();
    }

    /**
     * Get feature dimension
     */
    [[nodiscard]] size_t featureDim() const
    {
        return dataset_->featureDim();
    }

    /**
     * Get label dimension
     */
    [[nodiscard]] size_t labelDim() const
    {
        return dataset_->labelDim();
    }

    /**
     * Set whether to shuffle the data
     */
    void setShuffle(bool shuffle)
    {
        shuffle_ = shuffle;
        if (shuffle_)
        {
            shuffleIndices();
        }
    }

    /**
     * Set whether to drop the last incomplete batch
     */
    void setDropLast(bool dropLast)
    {
        dropLast_ = dropLast;
    }

    /**
     * Get feature names if available
     */
    [[nodiscard]] ds::Vector<ds::String> featureNames() const
    {
        return dataset_->featureNames();
    }

    /**
     * Get label names if available
     */
    [[nodiscard]] ds::Vector<ds::String> labelNames() const
    {
        return dataset_->labelNames();
    }

    /**
     * Get dataset description if available
     */
    [[nodiscard]] ds::String description() const
    {
        return dataset_->description();
    }

  private:
    std::shared_ptr<Dataset<T>> dataset_;
    size_t batchSize_;
    bool shuffle_;
    bool dropLast_;
    size_t currentIndex_;
    ds::Vector<size_t> indices_;
    std::random_device rd_;
    std::mt19937 gen_{rd_()};

    /**
     * Shuffle the indices
     */
    void shuffleIndices()
    {
        indices_.clear();
        indices_.reserve(dataset_->size());
        for (size_t i = 0; i < dataset_->size(); ++i)
        {
            indices_.pushBack(i);
        }
        std::ranges::shuffle(indices_, gen_);
    }
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_DATALOADER_H
