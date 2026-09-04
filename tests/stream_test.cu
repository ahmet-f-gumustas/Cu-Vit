#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/host_buffer.hpp"
#include "cuvit/stream.hpp"
#include "cuvit/vector_add.hpp"
#include "reference/elementwise.hpp"
#include "test_utils.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cuvit::testing::DataGenerator;
using cuvit::testing::require;

void test_stream_lifetime() {
    cuvit::Stream stream;
    require(!stream.empty(), "default-constructed stream is empty");
    require(stream.get() != nullptr, "default-constructed stream has no handle");

    const cudaStream_t handle = stream.get();
    cuvit::Stream moved(std::move(stream));
    require(stream.empty(), "moved-from stream still owns a handle");
    require(moved.get() == handle, "move did not transfer the handle");

    cuvit::Stream destination;
    const cudaStream_t replaced = destination.get();
    destination = std::move(moved);
    require(moved.empty(), "move-assigned source still owns a handle");
    require(destination.get() == handle, "move assignment did not transfer the handle");
    require(destination.get() != replaced, "move assignment kept the old handle");

    destination.synchronize();
    require(destination.query(), "an idle stream did not report completion");
}

// Runs the whole host-to-device, compute, device-to-host path on a non-default
// stream using pinned staging buffers, which is the shape the inference path
// will use.
void test_async_pipeline() {
    constexpr std::size_t element_count = 4099;

    cuvit::Stream stream;

    cuvit::HostBuffer<float> left(element_count);
    cuvit::HostBuffer<float> right(element_count);
    cuvit::HostBuffer<float> output(element_count);
    std::vector<float> expected(element_count);

    DataGenerator generator(7);
    generator.normal(left.data(), element_count);
    generator.normal(right.data(), element_count);
    cuvit::reference::vector_add(left.data(), right.data(), expected.data(), element_count);
    output.fill_zero();

    cuvit::DeviceBuffer<float> device_left(element_count);
    cuvit::DeviceBuffer<float> device_right(element_count);
    cuvit::DeviceBuffer<float> device_output(element_count);

    device_output.fill_zero_async(stream.get());
    device_left.copy_from_host_async(left.data(), left.size(), stream.get());
    device_right.copy_from_host_async(right.data(), right.size(), stream.get());

    cuvit::launch_vector_add(device_left.data(), device_right.data(), device_output.data(),
                             element_count, stream.get());

    device_output.copy_to_host_async(output.data(), output.size(), stream.get());
    stream.synchronize();

    cuvit::testing::require_exact(output.data(), expected.data(), element_count,
                                  "async pipeline result differs from the CPU reference");
}

void test_async_device_to_device() {
    constexpr std::size_t element_count = 128;

    cuvit::Stream stream;

    cuvit::HostBuffer<int> input(element_count);
    cuvit::HostBuffer<int> output(element_count);
    DataGenerator(11).integers(input.data(), element_count, -1000, 1000);
    output.fill_zero();

    cuvit::DeviceBuffer<int> source(element_count);
    cuvit::DeviceBuffer<int> destination(element_count);

    source.copy_from_host_async(input.data(), input.size(), stream.get());
    destination.copy_from_device_async(source, element_count, stream.get());
    destination.copy_to_host_async(output.data(), output.size(), stream.get());
    stream.synchronize();

    cuvit::testing::require_exact(output.data(), input.data(), element_count,
                                  "async device-to-device copy changed the data");

    bool overlap_error_seen = false;
    try {
        destination.copy_from_device_async(destination, 8, stream.get(), 0, 4);
    } catch (const std::invalid_argument&) {
        overlap_error_seen = true;
    }
    require(overlap_error_seen, "overlapping async device-to-device copy did not throw");
}

void test_streams_are_independent() {
    constexpr std::size_t element_count = 1024;

    cuvit::Stream first;
    cuvit::Stream second;
    require(first.get() != second.get(), "two streams share a handle");

    cuvit::HostBuffer<float> host(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        host[index] = static_cast<float>(index);
    }

    cuvit::DeviceBuffer<float> a(element_count);
    cuvit::DeviceBuffer<float> b(element_count);

    a.copy_from_host_async(host.data(), host.size(), first.get());
    b.copy_from_host_async(host.data(), host.size(), second.get());

    first.synchronize();
    second.synchronize();

    cuvit::HostBuffer<float> readback(element_count);
    readback.fill_zero();
    b.copy_to_host(readback.data(), readback.size());
    cuvit::testing::require_exact(readback.data(), host.data(), element_count,
                                  "work queued on a second stream did not complete");
}

void test_buffer_reports_its_device() {
    int active_device = -1;
    CUVIT_CUDA_CHECK(cudaGetDevice(&active_device));

    cuvit::DeviceBuffer<float> buffer(32);
    require(buffer.device() == active_device, "buffer did not record the active device");

    cuvit::DeviceBuffer<float> moved(std::move(buffer));
    require(buffer.device() == -1, "moved-from buffer still claims a device");
    require(moved.device() == active_device, "move lost the device identity");

    moved.reset();
    require(moved.device() == -1, "reset buffer still claims a device");
}

} // namespace

static_assert(!std::is_copy_constructible<cuvit::Stream>::value, "stream must be move-only");
static_assert(std::is_nothrow_move_constructible<cuvit::Stream>::value,
              "stream move must be noexcept");

CUVIT_TEST_MAIN("stream_test", {
    test_stream_lifetime();
    test_async_pipeline();
    test_async_device_to_device();
    test_streams_are_independent();
    test_buffer_reports_its_device();
})
