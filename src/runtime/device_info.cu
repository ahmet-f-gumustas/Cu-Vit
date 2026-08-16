#include "cuvit/device_info.hpp"

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

namespace cuvit {

DeviceInfo current_device_info() {
    int device_index = 0;
    CUVIT_CUDA_CHECK(cudaGetDevice(&device_index));

    cudaDeviceProp properties{};
    CUVIT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));

    DeviceInfo info;
    info.index = device_index;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.total_memory_bytes = properties.totalGlobalMem;
    return info;
}

} // namespace cuvit
