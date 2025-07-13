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
// Created by napbad on 7/12/25.
//

#ifndef TENSORSHAPE_H
#define TENSORSHAPE_H
#include "include/hiahiahia/common/defines.h"


namespace hiahiahia {
    class TensorShape {

    public:
        [[nodiscard]] uint len() const { return _len; }
        [[nodiscard]] uint *sizes() const { return _sizes; }

        // Constructor
        TensorShape(const uint *sizes, const uint len) {
            this->_len = len;
            this->_sizes = new uint[len];
            for (uint i = 0; i < len; ++i)
            {
                this->_sizes[i] = sizes[i];
            }
        }

        ~TensorShape() {
            delete[] _sizes;
        }

        [[nodiscard]] uint getTotalSize() const {
            uint res = 1;
            for (int i = 0; i < _len; ++i) {
                res *= _sizes[i];
            }
            return res;
        }

    private:
        uint _len;
        uint *_sizes;
    };


}


#endif //TENSORSHAPE_H
