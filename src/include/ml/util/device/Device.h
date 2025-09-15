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

#ifndef DEVICE_H
#define DEVICE_H
#include "common/ds/str.h"

namespace hiahiahia {

  enum class DeviceType {
    CPU = 0,
    GPU = 1
  };

  using ds::Str;

  class Device {
  public:
    virtual ~Device() = default;

    [[nodiscard]] virtual DeviceType type() const = 0;

    [[nodiscard]] virtual int id() const = 0;

    [[nodiscard]] virtual Str toString() const = 0;

    [[nodiscard]] virtual bool isAvailable() const = 0;

    virtual void activate() const = 0;

    bool operator==(const Device& other) const {
      return type() == other.type() && id() == other.id();
    }

    bool operator!=(const Device& other) const {
      return !(*this == other);
    }
  };

} // namespace hiahiahia

#endif //DEVICE_H
