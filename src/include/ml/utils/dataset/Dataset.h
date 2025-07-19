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
// Created by root on 8/3/25.
//

#ifndef DATASET_H
#define DATASET_H
#include <functional>
#include <mutex>

#include "DatasetIterator.h"
#include "include/common/defines/h3defs.h"

namespace hiahiahia {

  template<class SampleType>
  class Dataset {
  public:
    using Sample = SampleType;
    using Transform = std::function<Sample(const Sample&)>;
    using Iterator = DatasetIterator<Sample>;

    Dataset() = default;
    virtual ~Dataset() = default;

    Dataset(const Dataset &) = delete;
    Dataset& operator= (const Dataset &) = delete;
    Dataset(Dataset &&) = default;
    Dataset& operator= (Dataset &&) = default;

    [[nodiscard]] virtual sizeT size() const = 0;

    Sample operator()(const sizeT index) const {
      return get(index);
    }

    Iterator begin() const {
      return DatasetIterator<SampleType>(this, 0);
    }

    Iterator end() const {
      return Iterator(this, size());
    }

    virtual Sample get(sizeT index) const = 0;

    void set_transform(std::function<Sample(const Sample&)> transform) {
      std::lock_guard lock(_transform_mutex);
      _transform = std::move(transform);
    }

  private:
    mutable std::mutex _transform_mutex;
    std::function<Sample(const Sample&)> _transform;
  };

} // namespace hiahiahia

#endif //DATASET_H
