#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/vector_add.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void run_case(std::size_t element_count) {
    std::vector<float> left(element_count);
    std::vector<float> right(element_count);
    std::vector<float> output(element_count);

    for (std::size_t index = 0; index < element_count; ++index) {
        left[index] = std::sin(static_cast<float>(index) * 0.1F);
        right[index] = std::cos(static_cast<float>(index) * 0.2F);
    }

    cuvit::DeviceBuffer<float> device_left(element_count);
    cuvit::DeviceBuffer<float> device_right(element_count);
    cuvit::DeviceBuffer<float> device_output(element_count);

    device_left.copy_from_host(left.data(), element_count);
    device_right.copy_from_host(right.data(), element_count);

    cuvit::launch_vector_add(device_left.data(), device_right.data(), device_output.data(),
                             element_count);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), element_count);

    for (std::size_t index = 0; index < element_count; ++index) {
        const float expected = left[index] + right[index];
        if (std::abs(output[index] - expected) > 1.0e-6F) {
            throw std::runtime_error("vector-add result differs from the CPU reference");
        }
    }
}

} // namespace

int main() {
    try {
        cuvit::launch_vector_add(nullptr, nullptr, nullptr, 0);

        bool null_pointer_error_seen = false;
        try {
            cuvit::launch_vector_add(nullptr, nullptr, nullptr, 1);
        } catch (const std::invalid_argument&) {
            null_pointer_error_seen = true;
        }
        if (!null_pointer_error_seen) {
            throw std::runtime_error("non-empty vector-add accepted null pointers");
        }

        constexpr std::array<std::size_t, 7> sizes = {1, 31, 32, 255, 256, 257, 4099};
        for (const std::size_t size : sizes) {
            run_case(size);
        }

        std::cout << "vector_add_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vector_add_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
