# x86 SIMD configuration module
# Enables SSE2, AVX, AVX2 based on architecture and user preference

function(configure_x86_simd target_name)
    if (HAHAHA_ARCH_X86_64 OR HAHAHA_ARCH_X86_32)
        set(HAHAHA_SIMD_AVX OFF CACHE BOOL "Enable AVX/AVX2 for 256-bit SIMD (requires CPU support)")
        
        if (HAHAHA_SIMD_AVX)
            if (MSVC)
                target_compile_options(${target_name} PUBLIC /arch:AVX2)
            else ()
                target_compile_options(${target_name} PUBLIC -mavx2 -mfma)
            endif ()
        endif ()
        
        # SSE2 is default on x86-64; x86-32 may need -msse2
        if (HAHAHA_ARCH_X86_32 AND NOT MSVC)
            target_compile_options(${target_name} PUBLIC -msse2)
        endif ()
    endif ()
endfunction()
