#pragma once

#include <cstddef>
#include <string>

namespace cuvit {

struct DeviceInfo {
    int index = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_memory_bytes = 0;
};

[[nodiscard]] DeviceInfo current_device_info();

} // namespace cuvit
