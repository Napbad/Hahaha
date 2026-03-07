//  Copyright (c) 2025 - 2026 Contributors of
//  Hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//
//

#ifndef HAHAHA_MACROS_H_671376817DA64D9DBE364214872FCEE5
#define HAHAHA_MACROS_H_671376817DA64D9DBE364214872FCEE5

#ifdef HAHAHA_USE_CUDA
#define EnableWhileUseCuda(...) __VA_ARGS__
#else
#define EnableWhileUseCuda(...) // Expands to nothing
#endif

#ifdef HAHAHA_USE_CUDA
#define EnableWhileNoCuda(...) // Expands to nothing
#else
#define EnableWhileNoCuda(...) __VA_ARGS__
#endif

#if defined(_MSC_VER)
#define HAHAHA_PRAGMA_UNROLL(n)
#elif defined(__clang__)
#define HAHAHA_STR(x) #x
#define HAHAHA_XSTR(x) HAHAHA_STR(x)
#define HAHAHA_PRAGMA_UNROLL(n) _Pragma(HAHAHA_XSTR(unroll n))
#elif defined(__GNUC__)
#define HAHAHA_STR(x) #x
#define HAHAHA_XSTR(x) HAHAHA_STR(x)
#define HAHAHA_PRAGMA_UNROLL(n) _Pragma(HAHAHA_XSTR(GCC unroll n))
#else
#define HAHAHA_PRAGMA_UNROLL(n)
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)

    #ifndef ARCH_X86_64
        #define ARCH_X86_64 1
    #endif
    #ifndef ARCH_IS_X86_FAMILY
        #define ARCH_IS_X86_FAMILY 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 64
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "x86-64"
    #endif

#elif defined(__i386__) || defined(_M_IX86) || defined(__i386)

    #ifndef ARCH_X86_32
        #define ARCH_X86_32 1
    #endif
    #ifndef ARCH_IS_X86_FAMILY
        #define ARCH_IS_X86_FAMILY 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 32
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "x86"
    #endif

#elif defined(__aarch64__) || defined(_M_ARM64)

    #ifndef ARCH_ARM64
        #define ARCH_ARM64 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 64
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "arm64"
    #endif

#elif defined(__arm__) || defined(__thumb__) || defined(ARM) || defined(_M_ARM)

    #ifndef ARCH_ARM32
        #define ARCH_ARM32 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 32
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "arm32"
    #endif

#elif defined(__powerpc64__) || defined(__ppc64__)

    #ifndef ARCH_PPC64
        #define ARCH_PPC64 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 64
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "ppc64"
    #endif

#elif defined(__riscv) && __riscv_xlen == 64

    #ifndef ARCH_RISCV64
        #define ARCH_RISCV64 1
    #endif
    #ifndef ARCH_BITS
        #define ARCH_BITS 64
    #endif
    #ifndef ARCH_NAME
        #define ARCH_NAME "riscv64"
    #endif

#else

    #warning "Unknown / unsupported architecture"

    #ifndef ARCH_UNKNOWN
        #define ARCH_UNKNOWN 1
    #endif

#endif

#endif // HAHAHA_MACROS_H_671376817DA64D9DBE364214872FCEE5
