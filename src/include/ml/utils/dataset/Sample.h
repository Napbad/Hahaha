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
// Created by root on 8/4/25.
//

#ifndef SAMPLE_H
#define SAMPLE_H
#include "include/common/ds/str.h"
#include "include/common/ds/vec.h"
#include "include/common/util/commonUtil.h"

namespace hiahiahia {
  template<typename SampleType>
  class Sample {
  public:
    explicit Sample(const ds::Str &str, const char split = ',') {
      sizeT currBegin = 0;

      ds::Str curr;
      for (sizeT i = 0; i < str.size(); i++) {
        if (str[i] == split) {
          curr.reserve(i - currBegin);
          for (sizeT j = currBegin; j < i; j++) {
            curr.push_back(str[j]);
          }
          _data.emplace_back(util::strTo<SampleType>(curr));
        }
        currBegin = i + 1;
      }
    }

  private:
    ds::vec<SampleType> _data;
  };
} // namespace hiahiahia


#endif //SAMPLE_H
