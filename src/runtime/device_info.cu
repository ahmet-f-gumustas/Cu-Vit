#include "cuvit/device_info.hpp"

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace cuvit {
namespace {

// cudaGetDeviceProperties costs roughly a millisecond and the properties of a
// given device never change, so query each device at most once per process.
const DeviceInfo& cached_device_info(int device_index) {
    static std::mutex mutex;
    static std::unordered_map<int, DeviceInfo> cache;

    const std::lock_guard<std::mutex> lock(mutex);

    const auto existing = cache.find(device_index);
    if (existing != cache.end()) {
        return existing->second;
    }

    cudaDeviceProp properties{};
    CUVIT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));

    DeviceInfo info;
    info.index = device_index;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.total_memory_bytes = properties.totalGlobalMem;

    // References into an unordered_map stay valid across later insertions.
    return cache.emplace(device_index, std::move(info)).first->second;
}

} // namespace

DeviceInfo current_device_info() {
    int device_index = 0;
    CUVIT_CUDA_CHECK(cudaGetDevice(&device_index));
    return cached_device_info(device_index);
}

} // namespace cuvit
