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
// Created by Napbad on 7/12/25.
//

#ifndef MEMORY_H
#define MEMORY_H
#include "common/defines/h3defs.h"
#include "ml/common/defines.h"

namespace hahaha {
  inline void *hmalloc(uint size) { return nullptr; }

  inline void hfree(const void *ptr) { delete ptr; }

  inline void hmemcpy(void *src, void *dst, uint size) {

    if (src == dst) {
      return;
    }

    for (uint i = 0; i < size; ++i) {
      static_cast<char *>(dst)[i] = static_cast<char *>(src)[i];
    }
  }

} // namespace hahaha

#endif // MEMORY_H
