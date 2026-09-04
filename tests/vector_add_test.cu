#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/vector_add.hpp"
#include "reference/elementwise.hpp"
#include "test_utils.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using cuvit::testing::DataGenerator;
using cuvit::testing::require;

void run_case(std::size_t element_count) {
    std::vector<float> left(element_count);
    std::vector<float> right(element_count);
    std::vector<float> expected(element_count);
    std::vector<float> output(element_count);

    DataGenerator generator(element_count);
    generator.normal(left.data(), element_count);
    generator.normal(right.data(), element_count);
    cuvit::reference::vector_add(left.data(), right.data(), expected.data(), element_count);

    cuvit::DeviceBuffer<float> device_left(element_count);
    cuvit::DeviceBuffer<float> device_right(element_count);
    cuvit::DeviceBuffer<float> device_output(element_count);

    device_left.copy_from_host(left.data(), element_count);
    device_right.copy_from_host(right.data(), element_count);

    cuvit::launch_vector_add(device_left.data(), device_right.data(), device_output.data(),
                             element_count);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), element_count);

    // A single addition rounds identically on both sides, so this kernel is held
    // to bitwise equality. Kernels that reduce over many terms will not be.
    cuvit::testing::require_exact(output.data(), expected.data(), element_count,
                                  "vector-add result differs from the CPU reference");
}

void test_empty_launch_is_accepted() { cuvit::launch_vector_add(nullptr, nullptr, nullptr, 0); }

void test_null_pointers_are_rejected() {
    bool null_pointer_error_seen = false;
    try {
        cuvit::launch_vector_add(nullptr, nullptr, nullptr, 1);
    } catch (const std::invalid_argument&) {
        null_pointer_error_seen = true;
    }
    require(null_pointer_error_seen, "non-empty vector-add accepted null pointers");
}

void test_boundary_sizes() {
    constexpr std::array<std::size_t, 7> sizes = {1, 31, 32, 255, 256, 257, 4099};
    for (const std::size_t size : sizes) {
        run_case(size);
    }
}

} // namespace

CUVIT_TEST_MAIN("vector_add_test", {
    test_empty_launch_is_accepted();
    test_null_pointers_are_rejected();
    test_boundary_sizes();
})
