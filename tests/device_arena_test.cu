#include "cuvit/device_arena.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/tensor.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using cuvit::DeviceArena;
using cuvit::Extent;
using cuvit::TensorView;
using cuvit::testing::require;

bool is_aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

void test_capacity_accounting() {
    DeviceArena arena(4096);
    require(arena.capacity() == 4096, "arena has the wrong capacity");
    require(arena.used() == 0, "a fresh arena is not empty");
    require(arena.remaining() == 4096, "a fresh arena reports the wrong headroom");

    const TensorView<float> first = arena.allocate<float>({10});
    // 40 bytes of payload still consumes a full alignment block, because the
    // next tensor has to start aligned.
    require(arena.used() == DeviceArena::kAlignment, "the first tensor did not consume a block");
    require(first.size() == 10, "the tensor has the wrong element count");

    static_cast<void>(arena.allocate<float>({64}));
    require(arena.used() == DeviceArena::kAlignment + 256, "the second tensor was mis-accounted");

    arena.reset();
    require(arena.used() == 0, "reset did not rewind the arena");
    require(arena.capacity() == 4096, "reset released the allocation");
}

void test_every_tensor_is_aligned() {
    DeviceArena arena(1 << 16);
    // Sizes chosen to be awkward: none is a multiple of the alignment, so an
    // arena that simply bumped by the payload size would drift out of alignment.
    for (const Extent count : {1, 3, 17, 65, 129, 5}) {
        const TensorView<float> view = arena.allocate<float>({count});
        require(is_aligned(view.data(), DeviceArena::kAlignment),
                "an arena tensor is not aligned to the arena's guarantee");
    }
}

void test_tensors_do_not_overlap() {
    constexpr Extent count = 64;
    DeviceArena arena(1 << 16);

    TensorView<float> first = arena.allocate<float>({count});
    TensorView<float> second = arena.allocate<float>({count});
    require(second.data() >= first.data() + count, "two arena tensors overlap");

    // Writing through one view must leave the other untouched, which is the
    // property the whole arena rests on.
    std::vector<float> ones(count, 1.0F);
    std::vector<float> twos(count, 2.0F);
    std::vector<float> readback(count, 0.0F);

    CUVIT_CUDA_CHECK(
        cudaMemcpy(first.data(), ones.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUVIT_CUDA_CHECK(
        cudaMemcpy(second.data(), twos.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUVIT_CUDA_CHECK(
        cudaMemcpy(readback.data(), first.data(), count * sizeof(float), cudaMemcpyDeviceToHost));

    cuvit::testing::require_exact(readback.data(), ones.data(), count,
                                  "writing the second tensor corrupted the first");
}

void test_exhaustion_is_reported() {
    DeviceArena arena(1024);
    static_cast<void>(arena.allocate<float>({128})); // 512 bytes

    bool caught = false;
    try {
        static_cast<void>(arena.allocate<float>({256})); // 1024 bytes, more than remains
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught, "an over-capacity allocation was accepted");

    // The failed request must not have consumed anything, so a caller that
    // handles the error can keep using the arena.
    require(arena.used() == 512, "a failed allocation moved the arena cursor");
    static_cast<void>(arena.allocate<float>({128}));
    require(arena.used() == 1024, "the arena was unusable after a failed allocation");
}

void test_reserved_bytes_predicts_usage() {
    // Sizing an arena before creating it has to be exact, or a weight load fails
    // partway through with the tensors already scattered.
    const std::vector<Extent> counts = {1, 100, 192, 768, 147456};
    std::size_t predicted = 0;
    for (const Extent count : counts) {
        predicted += DeviceArena::reserved_bytes<float>(static_cast<std::size_t>(count));
    }

    DeviceArena arena(predicted);
    for (const Extent count : counts) {
        static_cast<void>(arena.allocate<float>({count}));
    }
    require(arena.used() == predicted, "reserved_bytes disagrees with actual usage");
    require(arena.remaining() == 0, "the arena was sized larger than needed");
}

void test_zero_fill_covers_the_whole_arena() {
    constexpr Extent count = 32;
    DeviceArena arena(4096);
    TensorView<float> view = arena.allocate<float>({count});

    std::vector<float> noise(count, 7.0F);
    CUVIT_CUDA_CHECK(
        cudaMemcpy(view.data(), noise.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    arena.fill_zero();

    std::vector<float> readback(count, -1.0F);
    CUVIT_CUDA_CHECK(
        cudaMemcpy(readback.data(), view.data(), count * sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<float> zeros(count, 0.0F);
    cuvit::testing::require_exact(readback.data(), zeros.data(), count,
                                  "fill_zero left non-zero values in the arena");
}

void test_multidimensional_allocation() {
    DeviceArena arena(1 << 20);
    const TensorView<float> view = arena.allocate<float>({2, 197, 192});
    require(view.rank() == 3, "arena tensor has the wrong rank");
    require(view.size() == 2 * 197 * 192, "arena tensor has the wrong element count");
    require(view.is_contiguous(), "an arena tensor is not contiguous");
    require(arena.used() == DeviceArena::reserved_bytes<float>(2 * 197 * 192),
            "a multidimensional tensor was mis-accounted");
}

} // namespace

CUVIT_TEST_MAIN("device_arena_test", {
    test_capacity_accounting();
    test_every_tensor_is_aligned();
    test_tensors_do_not_overlap();
    test_exhaustion_is_reported();
    test_reserved_bytes_predicts_usage();
    test_zero_fill_covers_the_whole_arena();
    test_multidimensional_allocation();
})
