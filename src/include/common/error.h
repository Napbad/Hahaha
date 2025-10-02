/*
 * @Author: napbad sen napbad.sen@gmail.com
 * @Date: 2025-10-02 14:23:30
 * @LastEditors: napbad sen napbad.sen@gmail.com
 * @LastEditTime: 2025-10-02 22:43:56
 * @FilePath: /hahaha-dev/src/include/common/Error.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
// Created by Napbad on 7/27/25.
//

#ifndef ERR_H
#define ERR_H
#include "ds/Str.h"

namespace hahaha::common {
    namespace ds {
        class Str;
    }
    using ds::Str;

    class error {
    public:
        virtual ~error() = default;

        // Get the error type name
        [[nodiscard]] virtual Str typeName() const = 0;

        // Get the error message
        [[nodiscard]] virtual Str message() const = 0;

        // Get the error location
        [[nodiscard]] virtual Str location() const = 0;

        // Convert the error to a string
        [[nodiscard]] virtual Str toString() const = 0;

        error() = default;
    };

    class BaseErr : public error {
    public:
        explicit BaseErr(Str msg);
        explicit BaseErr(Str msg, Str loc);
        explicit BaseErr(const char* msg);
        explicit BaseErr(const char* msg, const char* loc);

        BaseErr();
        ~BaseErr() override = default;

        [[nodiscard]] Str typeName() const override;
        [[nodiscard]] Str message() const override;
        [[nodiscard]] Str location() const override;
        [[nodiscard]] Str toString() const override;

    private:
        Str _msg;
        Str _loc;
        const Str TypeName = Str("BaseError");
    };

    class IndexOutOfBoundError final : public BaseErr {
    public:
        explicit IndexOutOfBoundError(Str msg) : BaseErr(std::move(msg), Str("IndexOutOfBoundError")) {}
        explicit IndexOutOfBoundError(Str msg, Str loc) : BaseErr(std::move(msg), std::move(loc)) {}
        explicit IndexOutOfBoundError(const char* msg) : BaseErr(Str(msg), Str("IndexOutOfBoundError")) {}
        explicit IndexOutOfBoundError(const char* msg, const char* loc) : BaseErr(Str(msg), Str(loc)) {}

        [[nodiscard]] Str typeName() const override { return Str("IndexOutOfBoundError"); }
    };

    // Add other common error types here as needed

} // namespace hahaha::common

#endif // ERR_H
