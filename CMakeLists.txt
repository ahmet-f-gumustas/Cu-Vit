cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(
    CuVit
    VERSION 0.3.1
    DESCRIPTION "A Vision Transformer inference engine implemented in CUDA C++"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

option(CUVIT_BUILD_TESTS "Build the Cu-Vit test suite" ON)

function(cuvit_apply_common_options target_name)
    set(cuvit_debug_config "$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>")

    target_compile_options(${target_name}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
            # Source correlation for compute-sanitizer and Nsight.
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,${cuvit_debug_config}>:-lineinfo>"
    )
endfunction()

add_library(cuvit STATIC
    src/kernels/gemm.cu
    src/kernels/normalization.cu
    src/kernels/vector_add.cu
    src/runtime/device_info.cu
    src/model/vit.cu
    src/runtime/image.cpp
    src/runtime/weights.cpp
)

target_include_directories(cuvit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cuvit PUBLIC CUDA::cudart)
cuvit_apply_common_options(cuvit)

set_target_properties(cuvit PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_executable(cu-vit src/main.cpp)
target_link_libraries(cu-vit PRIVATE cuvit)
cuvit_apply_common_options(cu-vit)

if(CUVIT_BUILD_TESTS)
    enable_testing()

    set(cuvit_test_names
        device_arena_test
        device_buffer_test
        elementwise_test
        gemm_test
        image_test
        host_buffer_test
        numerics_test
        stream_test
        tensor_test
        vector_add_test
        vit_test
        weights_test
    )

    foreach(cuvit_test_name IN LISTS cuvit_test_names)
        add_executable(${cuvit_test_name} tests/${cuvit_test_name}.cu)
        target_include_directories(${cuvit_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(${cuvit_test_name} PRIVATE cuvit)
        cuvit_apply_common_options(${cuvit_test_name})
        add_test(NAME ${cuvit_test_name} COMMAND ${cuvit_test_name})
    endforeach()
endif()
