#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/host_buffer.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool is_pinned(const void* pointer) {
    cudaPointerAttributes attributes{};
    CUVIT_CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    return attributes.type == cudaMemoryTypeHost;
}

void test_allocation_is_page_locked() {
    constexpr std::size_t element_count = 512;

    cuvit::HostBuffer<float> buffer(element_count);
    require(buffer.size() == element_count, "allocation has the wrong element count");
    require(buffer.bytes() == element_count * sizeof(float), "allocation has the wrong byte count");
    require(!buffer.empty(), "allocated buffer is unexpectedly empty");
    require(is_pinned(buffer.data()), "HostBuffer memory is not page-locked");

    // A plain heap allocation must not be mistaken for pinned memory, otherwise
    // the check above would pass for any pointer.
    float pageable = 0.0F;
    require(!is_pinned(&pageable), "pageable memory was reported as page-locked");
}

void test_element_access_and_fill() {
    constexpr std::size_t element_count = 64;

    cuvit::HostBuffer<int> buffer(element_count);
    for (std::size_t index = 0; index < buffer.size(); ++index) {
        buffer[index] = static_cast<int>(index) * 7;
    }

    require(buffer[0] == 0 && buffer[63] == 63 * 7, "element write did not survive a read back");
    require(std::equal(buffer.begin(), buffer.end(), buffer.data()),
            "iterators disagree with data()");

    bool range_error_seen = false;
    try {
        static_cast<void>(buffer.at(element_count));
    } catch (const std::out_of_range&) {
        range_error_seen = true;
    }
    require(range_error_seen, "out-of-range at() did not throw");

    buffer.fill_zero();
    require(std::all_of(buffer.begin(), buffer.end(), [](int value) { return value == 0; }),
            "zero fill left non-zero values");
}

void test_allocate_keeps_contents_at_the_same_size() {
    cuvit::HostBuffer<int> buffer(4);
    buffer.fill_zero();
    buffer[2] = 99;

    buffer.allocate(4);
    require(buffer[2] == 99, "same-size allocate discarded the existing contents");

    buffer.allocate(8);
    require(buffer.size() == 8, "resize did not change the element count");
}

void test_move_semantics() {
    cuvit::HostBuffer<double> source(16);
    source[0] = 3.5;
    const double* const original = source.data();

    cuvit::HostBuffer<double> moved(std::move(source));
    require(source.empty(), "moved-from buffer still owns an allocation");
    require(source.data() == nullptr, "moved-from buffer still exposes a pointer");
    require(moved.data() == original, "move reallocated instead of transferring ownership");
    require(moved[0] == 3.5, "move lost the contents");

    cuvit::HostBuffer<double> destination(2);
    destination = std::move(moved);
    require(moved.empty(), "move-assigned source still owns an allocation");
    require(destination.size() == 16, "move assignment lost the allocation");
    require(destination[0] == 3.5, "move assignment lost the contents");
}

void test_round_trip_through_device() {
    constexpr std::size_t element_count = 257;

    cuvit::HostBuffer<float> input(element_count);
    cuvit::HostBuffer<float> output(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        input[index] = static_cast<float>(index) * 0.5F;
    }
    output.fill_zero();

    cuvit::DeviceBuffer<float> device(element_count);
    device.copy_from_host(input.data(), input.size());
    device.copy_to_host(output.data(), output.size());

    require(std::equal(input.begin(), input.end(), output.begin()),
            "pinned host round trip changed the data");
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible<cuvit::HostBuffer<float>>::value,
                  "buffer must be move-only");
    static_assert(std::is_nothrow_move_constructible<cuvit::HostBuffer<float>>::value,
                  "buffer move must be noexcept");

    try {
        test_allocation_is_page_locked();
        test_element_access_and_fill();
        test_allocate_keeps_contents_at_the_same_size();
        test_move_semantics();
        test_round_trip_through_device();
        CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "host_buffer_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host_buffer_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
