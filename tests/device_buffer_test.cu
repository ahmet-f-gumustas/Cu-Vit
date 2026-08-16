#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_allocation_and_copy() {
    constexpr std::size_t element_count = 1024;
    std::vector<int> input(element_count);
    std::vector<int> output(element_count, 0);

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<int>(index * 3);
    }

    cuvit::DeviceBuffer<int> buffer(element_count);
    require(buffer.size() == element_count, "allocation has the wrong element count");
    require(buffer.bytes() == element_count * sizeof(int), "allocation has the wrong byte count");
    require(!buffer.empty(), "allocated buffer is unexpectedly empty");

    buffer.copy_from_host(input.data(), input.size());
    buffer.copy_to_host(output.data(), output.size());
    require(input == output, "host-to-device round trip changed the data");

    cuvit::DeviceBuffer<int> copy(element_count + 2);
    copy.fill_zero();
    copy.copy_from_device(buffer, element_count, 0, 1);

    std::vector<int> copied(element_count + 2, -1);
    copy.copy_to_host(copied.data(), copied.size());
    require(copied.front() == 0 && copied.back() == 0, "device copy overwrote guard elements");
    require(std::equal(input.begin(), input.end(), copied.begin() + 1),
            "device-to-device copy changed the data");
}

void test_move_and_zero_fill() {
    constexpr std::size_t element_count = 257;
    std::vector<float> input(element_count, 42.0F);
    std::vector<float> output(element_count, 1.0F);

    cuvit::DeviceBuffer<float> source(element_count);
    source.copy_from_host(input.data(), input.size());

    cuvit::DeviceBuffer<float> moved(std::move(source));
    require(source.empty(), "moved-from buffer still owns an allocation");
    require(moved.size() == element_count, "moved buffer lost its element count");

    cuvit::DeviceBuffer<float> destination(3);
    destination = std::move(moved);
    require(moved.empty(), "move-assigned source still owns an allocation");
    require(destination.size() == element_count, "move assignment lost the allocation");

    destination.fill_zero();
    destination.copy_to_host(output.data(), output.size());
    require(std::all_of(output.begin(), output.end(), [](float value) { return value == 0.0F; }),
            "zero fill left non-zero values in device memory");
}

void test_bounds_and_cuda_errors() {
    cuvit::DeviceBuffer<int> buffer(4);
    const int input[5] = {1, 2, 3, 4, 5};

    bool bounds_error_seen = false;
    try {
        buffer.copy_from_host(input, 5);
    } catch (const std::out_of_range&) {
        bounds_error_seen = true;
    }
    require(bounds_error_seen, "out-of-bounds copy did not throw");

    bool cuda_error_seen = false;
    try {
        CUVIT_CUDA_CHECK(cudaErrorInvalidValue);
    } catch (const cuvit::CudaError& error) {
        cuda_error_seen =
            error.code() == cudaErrorInvalidValue &&
            std::string(error.what()).find("cudaErrorInvalidValue") != std::string::npos;
    }
    require(cuda_error_seen, "CUDA error context was not preserved");
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible<cuvit::DeviceBuffer<float>>::value,
                  "buffer must be move-only");
    static_assert(std::is_nothrow_move_constructible<cuvit::DeviceBuffer<float>>::value,
                  "buffer move must be noexcept");

    try {
        test_allocation_and_copy();
        test_move_and_zero_fill();
        test_bounds_and_cuda_errors();
        CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "device_buffer_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "device_buffer_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
