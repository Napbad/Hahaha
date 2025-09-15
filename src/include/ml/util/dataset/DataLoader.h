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

#include <common/ds/Vec.h>
#include <common/Res.h>
#include <common/Error.h>
#include <ml/util/dataset/Dataset.h>
#include <ml/util/dataset/Sample.h>
#include <memory>
#include <random>

namespace hiahiahia::ml {

  /**
   * DataLoader error class
   */
  class DataLoaderError : public Error {
public:
    explicit DataLoaderError(ds::Str message, ds::Str location = ds::Str("DataLoader")) :
        _message(std::move(message)), _location(std::move(location)) {}

    [[nodiscard]] ds::Str typeName() const override { return ds::Str("DataLoaderError"); }
    [[nodiscard]] ds::Str message() const override { return _message; }
    [[nodiscard]] ds::Str location() const override { return _location; }
    [[nodiscard]] ds::Str toString() const override {
      return typeName() + ds::Str(": ") + message() + ds::Str(" at ") + location();
    }

private:
    ds::Str _message;
    ds::Str _location;
  };

  /**
   * DataLoader class
   *
   * Provides functionality to load data in batches with optional shuffling
   * and preprocessing capabilities.
   */
  template<typename T>
  class DataLoader {
public:
    /**
     * Create a new DataLoader
     */
    DataLoader(std::shared_ptr<Dataset<T>> dataset, size_t batchSize, bool shuffle = true, bool dropLast = false) :
        _dataset(std::move(dataset)), _batchSize(batchSize), _shuffle(shuffle), _dropLast(dropLast), _currentIndex(0) {
      if (_shuffle) {
        shuffleIndices();
      } else {
        _indices.reserve(_dataset->size());
        for (size_t i = 0; i < _dataset->size(); ++i) {
          _indices.push_back(i);
        }
      }
    }

    /**
     * Reset the iterator to the beginning and optionally reshuffle
     */
    void reset() {
      _currentIndex = 0;
      if (_shuffle) {
        shuffleIndices();
      }
    }

    /**
     * Get the next batch of samples
     */
    Res<ds::Vec<Sample<T>>, DataLoaderError> nextBatch() {
      SetRetT(ds::Vec<Sample<T>>, DataLoaderError);

      if (_currentIndex >= _dataset->size()) {
        Err(DataLoaderError(ds::Str("No more batches available")));
      }

      ds::Vec<Sample<T>> batch;
      const size_t remaining = std::min(_batchSize, _dataset->size() - _currentIndex);

      // If drop_last is true and this is the last incomplete batch, return error
      if (_dropLast && remaining < _batchSize) {
        Err(DataLoaderError(ds::Str("Dropping last incomplete batch")));
      }

      // Load the batch
      for (size_t i = 0; i < remaining; ++i) {
        auto sampleRes = _dataset->get(_indices[_currentIndex + i]);
        if (sampleRes.isErr()) {
          Err(DataLoaderError(sampleRes.unwrapErr().message()));
        }
        batch.push_back(sampleRes.unwrap());
      }

      _currentIndex += remaining;
      Ok(std::move(batch));
    }

    /**
     * Check if there are more batches available
     */
    [[nodiscard]] bool hasNext() const {
      if (_dropLast) {
        return _currentIndex + _batchSize <= _dataset->size();
      }
      return _currentIndex < _dataset->size();
    }

    /**
     * Get the total number of batches
     */
    [[nodiscard]] size_t numBatches() const {
      if (_dropLast) {
        return _dataset->size() / _batchSize;
      }
      return (_dataset->size() + _batchSize - 1) / _batchSize;
    }

    /**
     * Get the batch size
     */
    [[nodiscard]] size_t batchSize() const { return _batchSize; }

    /**
     * Get the dataset size
     */
    [[nodiscard]] size_t datasetSize() const { return _dataset->size(); }

    /**
     * Get feature dimension
     */
    [[nodiscard]] size_t featureDim() const { return _dataset->featureDim(); }

    /**
     * Get label dimension
     */
    [[nodiscard]] size_t labelDim() const { return _dataset->labelDim(); }

    /**
     * Set whether to shuffle the data
     */
    void setShuffle(bool shuffle) {
      _shuffle = shuffle;
      if (_shuffle) {
        shuffleIndices();
      }
    }

    /**
     * Set whether to drop the last incomplete batch
     */
    void setDropLast(bool dropLast) { _dropLast = dropLast; }

    /**
     * Get feature names if available
     */
    [[nodiscard]] ds::Vec<ds::Str> featureNames() const { return _dataset->featureNames(); }

    /**
     * Get label names if available
     */
    [[nodiscard]] ds::Vec<ds::Str> labelNames() const { return _dataset->labelNames(); }

    /**
     * Get dataset description if available
     */
    [[nodiscard]] ds::Str description() const { return _dataset->description(); }

private:
    std::shared_ptr<Dataset<T>> _dataset;
    size_t _batchSize;
    bool _shuffle;
    bool _dropLast;
    size_t _currentIndex;
    ds::Vec<size_t> _indices;
    std::random_device _rd;
    std::mt19937 _gen{_rd()};

    /**
     * Shuffle the indices
     */
    void shuffleIndices() {
      _indices.clear();
      _indices.reserve(_dataset->size());
      for (size_t i = 0; i < _dataset->size(); ++i) {
        _indices.push_back(i);
      }
      std::ranges::shuffle(_indices, _gen);
    }
  };

} // namespace hiahiahia::ml

#endif // HIAHIAHIA_DATALOADER_H 