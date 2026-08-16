cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(
    CuVit
    VERSION 0.1.0
    DESCRIPTION "A Vision Transformer inference engine implemented in CUDA C++"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

option(CUVIT_BUILD_TESTS "Build the Cu-Vit test suite" ON)

function(cuvit_enable_warnings target_name)
    target_compile_options(${target_name}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    )
endfunction()

add_library(cuvit STATIC
    src/kernels/vector_add.cu
    src/runtime/device_info.cu
)

target_include_directories(cuvit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cuvit PUBLIC CUDA::cudart)
cuvit_enable_warnings(cuvit)

set_target_properties(cuvit PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(cu-vit src/main.cpp)
target_link_libraries(cu-vit PRIVATE cuvit)
cuvit_enable_warnings(cu-vit)

if(CUVIT_BUILD_TESTS)
    include(CTest)

    add_executable(device_buffer_test tests/device_buffer_test.cu)
    target_link_libraries(device_buffer_test PRIVATE cuvit)
    cuvit_enable_warnings(device_buffer_test)
    add_test(NAME device_buffer_test COMMAND device_buffer_test)

    add_executable(vector_add_test tests/vector_add_test.cu)
    target_link_libraries(vector_add_test PRIVATE cuvit)
    cuvit_enable_warnings(vector_add_test)
    add_test(NAME vector_add_test COMMAND vector_add_test)
endif()
