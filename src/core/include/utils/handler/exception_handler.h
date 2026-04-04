//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/29/26.
//

#ifndef HAHAHA_EXCEPTION_HANDLER_H_DBA81902DB894B6D944EA99780129F9C
#define HAHAHA_EXCEPTION_HANDLER_H_DBA81902DB894B6D944EA99780129F9C
#define ThrowWithFormat(ExceptionType, ...) \
throw ExceptionType(std::format(__VA_ARGS__))

// 1. For invalid arguments passed to a function
#define ThrowInvalid(...)    ThrowWithFormat(std::invalid_argument, __VA_ARGS__)

// 2. For errors that occur at runtime (files, networks, etc.)
#define ThrowRuntime(...)    ThrowWithFormat(std::runtime_error, __VA_ARGS__)

// 3. For index-out-of-bounds errors
#define ThrowOutOfRange(...) ThrowWithFormat(std::out_of_range, __VA_ARGS__)

// 4. For internal logic/pre-condition violations
#define ThrowLogic(...)      ThrowWithFormat(std::logic_error, __VA_ARGS__)

// 5. For resource/memory overflow (less common but useful)
#define ThrowOverflow(...)   ThrowWithFormat(std::overflow_error, __VA_ARGS__)

#endif //HAHAHA_EXCEPTION_HANDLER_H_DBA81902DB894B6D944EA99780129F9C