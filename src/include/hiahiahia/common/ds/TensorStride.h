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
// Created by root on 7/13/25.
//

#ifndef TENSORSTRIDE_H
#define TENSORSTRIDE_H
#include "include/hiahiahia/common/defines.h"

namespace hiahiahia
{
    class TensorStride
    {
    public:

        explicit TensorStride(const uint len)
        {
            _len = len;
            _stride = new uint[len];
        }

        TensorStride(const uint len, const uint *stride)
        {
            _len = len;
            _stride = new uint[len];
            for (uint i = 0; i < len; i++)
            {
                _stride[i] = stride[i];
            }
        }

        explicit TensorStride(const TensorShape & shape)
        {
            _len = shape.len();
            _stride = new uint[_len];
            calculateStride(shape.sizes(), shape.len());
        }

        void calculateStride(const uint* shape, const uint len) const
        {
            if (len != _len)
            {
                return;
            }
            uint currStride = 1;
            for (uint i = 0; i < len; ++i)
            {
                this->_stride[_len - i - 1] = currStride;
                currStride *= shape[_len - i - 1];
            }
        }
    private:
        uint _len{};
        uint *_stride;
    };

}

#endif //TENSORSTRIDE_H
