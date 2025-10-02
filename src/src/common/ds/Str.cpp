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
// Created by napbad on 10/2/25.
//

#include "common/ds/Str.h"

#include "common/Res.h"
#include "common/error.h"
namespace hahaha {
    class IndexOutOfBoundError;
}

namespace hahaha::common::ds {
    Res<char, IndexOutOfBoundError> Str::at(sizeT i) const {
        SetRetT(char, IndexOutOfBoundError)
        if (i >= _size) {
            Err("Index out of bound");
        }
        Ok(_data[i]);
    }
}

