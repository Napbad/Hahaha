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
// Created by Napbad on 7/26/25.
//

#ifndef ARR_H
#define ARR_H

namespace hiahiahia::ds {
    template<class T, int len>
    class arr {
    public:
        void set(int i, T val) {
            if (i < 0 || i >= len) {  }
            data[i] = val;
        }

        T get(int i) {
            return data[i];
        }

    private:
        T data[len];

    };
} // namespace hiahiahia


#endif //ARR_H
