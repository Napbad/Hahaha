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
// Created by Napbad on 10/4/25.
//

#ifndef HIAHIAHIA_TRAINSTATISTICS_H
#define HIAHIAHIA_TRAINSTATISTICS_H
#include "defines/h3defs.h"
#include "ds/map.h"
#include <ds/Vector.h>

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{

    class TrainStatistics
    {
      public:
        ds::Vector<f32> losses;
    };

    class EmptyTrainStatistics : public TrainStatistics
    {
    };

    class LossTrainStatistics : public TrainStatistics
    {
      public:
      private:
    };
}

#endif // HIAHIAHIA_TRAINSTATISTICS_H
