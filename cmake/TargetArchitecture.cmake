# Target architecture configuration module
# Applies architecture-specific compile definitions to targets

function(apply_architecture_to_target target_name)
    # Define architecture macros via compile definitions
    if (HAHAHA_ARCH_X86_64)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_X86_64=1
            HAHAHA_ARCH_IS_X86_FAMILY=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_X86_32)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_X86_32=1
            HAHAHA_ARCH_IS_X86_FAMILY=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_ARM64)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_ARM64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_ARM32)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_ARM32=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_PPC64)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_PPC64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_RISCV64)
        target_compile_definitions(${target_name} PUBLIC
            HAHAHA_ARCH_RISCV64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
        )
    elseif (HAHAHA_ARCH_UNKNOWN)
        target_compile_definitions(${target_name} PUBLIC HAHAHA_ARCH_UNKNOWN=1)
    endif ()
endfunction()
