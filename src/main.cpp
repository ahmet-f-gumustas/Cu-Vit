#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/device_info.hpp"
#include "cuvit/vector_add.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void run_smoke_test() {
    constexpr std::size_t element_count = 4096;

    std::vector<float> left(element_count);
    std::vector<float> right(element_count);
    std::vector<float> output(element_count);

    for (std::size_t index = 0; index < element_count; ++index) {
        left[index] = static_cast<float>(index) * 0.25F;
        right[index] = static_cast<float>(index % 17) - 8.0F;
    }

    cuvit::DeviceBuffer<float> device_left(element_count);
    cuvit::DeviceBuffer<float> device_right(element_count);
    cuvit::DeviceBuffer<float> device_output(element_count);

    device_left.copy_from_host(left.data(), left.size());
    device_right.copy_from_host(right.data(), right.size());

    cuvit::launch_vector_add(device_left.data(), device_right.data(), device_output.data(),
                             element_count);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(output.data(), output.size());
    for (std::size_t index = 0; index < element_count; ++index) {
        const float expected = left[index] + right[index];
        if (std::abs(output[index] - expected) > 1.0e-6F) {
            throw std::runtime_error("CUDA smoke test produced an incorrect result");
        }
    }
}

} // namespace

int main() {
    try {
        const cuvit::DeviceInfo device = cuvit::current_device_info();

        std::cout << "Cu-Vit\n"
                  << "GPU: " << device.name << '\n'
                  << "Compute capability: " << device.compute_major << '.' << device.compute_minor
                  << '\n'
                  << "Global memory: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(device.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB\n";

        run_smoke_test();
        std::cout << "CUDA smoke test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cu-vit failed: " << error.what() << '\n';
        return 1;
    }
}
