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
// Created by Napbad on 8/3/25.
//

#ifndef DATASETITERATOR_H
#define DATASETITERATOR_H
#include <iterator>


#include "common/defines/h3defs.h"
namespace hahaha {

  template<class SampleType>
  class Dataset;


  template<typename SampleType>
  class DatasetIterator {
public:
    using iteratorCategory = std::forward_iterator_tag;
    using valueType = SampleType;
    using differenceType = ptrDiffT;
    using pointer = const SampleType *;
    using reference = const SampleType &;

    DatasetIterator(const Dataset<SampleType> *dataset, size_t index) : _dataset(dataset), _index(index) {}

    reference operator*() const {
      if (_index >= _dataset->size()) {
        throw std::out_of_range("Dataset iterator out of range");
      }
      return _dataset->get(_index);
    }

    pointer operator->() const { return &operator*(); }

    DatasetIterator &operator++() {
      _index++;
      return *this;
    }

    DatasetIterator operator++(int) {
      DatasetIterator temp = *this;
      _index++;
      return temp;
    }

    bool operator==(const DatasetIterator &other) const { return _dataset == other._dataset && _index == other._index; }

    bool operator!=(const DatasetIterator &other) const { return !(*this == other); }

private:
    const Dataset<SampleType> *_dataset;
    size_t _index;
  };
} // namespace hahaha


#endif // DATASETITERATOR_H
