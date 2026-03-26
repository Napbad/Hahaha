# Library linking module
# Handles linking of system and external libraries

function(link_system_libraries target_name)
    # dl (Linux/Unix) - harmless elsewhere
    if (CMAKE_DL_LIBS)
        target_link_libraries(${target_name} PUBLIC ${CMAKE_DL_LIBS})
    endif ()

    # std::stacktrace may require extra libs on some gcc/libstdc++ toolchains
    foreach (_libname IN ITEMS stdc++_libbacktrace stdc++exp backtrace)
        find_library(HAHAHA_STACKTRACE_${_libname} NAMES ${_libname})
        if (HAHAHA_STACKTRACE_${_libname})
            target_link_libraries(${target_name} PUBLIC ${HAHAHA_STACKTRACE_${_libname}})
        endif ()
    endforeach ()
endfunction()


function(link_cuda_libraries target_name)
    if (HAHAHA_USE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${target_name} PUBLIC CUDA::cudart)
    endif ()
endfunction()
