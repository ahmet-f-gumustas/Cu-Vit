#pragma once

#include "cuvit/device_buffer.hpp"
#include "cuvit/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <string>

namespace cuvit {

/// A bump allocator over one device allocation.
///
/// ViT-Tiny/16 holds about 150 weight tensors. Giving each its own cudaMalloc
/// costs 150 driver round trips and scatters them across the heap; one arena
/// costs a single allocation and keeps related tensors adjacent. Suballocation
/// is a pointer bump, so it is also cheap enough to redo per inference for
/// activations.
///
/// Nothing is freed individually: reset() rewinds the whole arena at once, which
/// is the right model for a graph whose lifetimes all end together. Views handed
/// out before a reset dangle afterwards.
class DeviceArena final {
  public:
    /// Alignment applied to every suballocation.
    ///
    /// 256 bytes is the guarantee cudaMalloc itself provides and what the
    /// coalescing rules and cuBLAS's alignment-sensitive paths expect, so a
    /// tensor placed inside the arena behaves like one that was allocated alone.
    static constexpr std::size_t kAlignment = 256;

    DeviceArena() noexcept = default;

    explicit DeviceArena(std::size_t capacity_bytes) { reserve(capacity_bytes); }

    /// Replaces the backing allocation, discarding everything already handed out.
    void reserve(std::size_t capacity_bytes) {
        storage_.allocate(capacity_bytes);
        used_ = 0;
    }

    /// Rewinds to empty without releasing the allocation.
    void reset() noexcept { used_ = 0; }

    /// Bytes one tensor of @p count elements occupies, including its padding.
    /// Use it to size an arena before creating it.
    template <typename T> [[nodiscard]] static std::size_t reserved_bytes(std::size_t count) {
        return align_up(checked_bytes<T>(count));
    }

    /// Carves out a contiguous tensor and returns a view of it.
    ///
    /// The contents are undefined; call fill_zero() on the arena beforehand when
    /// a kernel reads memory it has not written.
    template <typename T>
    [[nodiscard]] TensorView<T> allocate(std::initializer_list<Extent> extents) {
        Extent count = 1;
        for (const Extent extent : extents) {
            if (extent < 0) {
                throw std::invalid_argument("DeviceArena extents cannot be negative");
            }
            count *= extent;
        }
        const std::size_t bytes = checked_bytes<T>(static_cast<std::size_t>(count));
        return TensorView<T>(static_cast<T*>(carve(bytes, alignof(T))), extents);
    }

    /// Carves out a raw span of @p count elements.
    template <typename T> [[nodiscard]] T* allocate_raw(std::size_t count) {
        return static_cast<T*>(carve(checked_bytes<T>(count), alignof(T)));
    }

    void fill_zero() { storage_.fill_zero(); }
    void fill_zero_async(cudaStream_t stream) { storage_.fill_zero_async(stream); }

    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t used() const noexcept { return used_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return capacity() - used_; }
    [[nodiscard]] const std::byte* data() const noexcept { return storage_.data(); }

  private:
    static constexpr std::size_t align_up(std::size_t bytes) noexcept {
        return (bytes + kAlignment - 1) / kAlignment * kAlignment;
    }

    template <typename T> static std::size_t checked_bytes(std::size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            throw std::overflow_error("DeviceArena allocation size overflow");
        }
        return count * sizeof(T);
    }

    void* carve(std::size_t bytes, std::size_t alignment) {
        if (alignment > kAlignment) {
            throw std::invalid_argument("DeviceArena cannot satisfy an over-aligned type");
        }
        const std::size_t padded = align_up(bytes);
        if (padded > remaining()) {
            throw std::runtime_error("DeviceArena is out of capacity: " + std::to_string(padded) +
                                     " bytes requested, " + std::to_string(remaining()) + " of " +
                                     std::to_string(capacity()) + " remaining");
        }
        std::byte* const pointer = storage_.data() + used_;
        used_ += padded;
        return pointer;
    }

    DeviceBuffer<std::byte> storage_;
    std::size_t used_ = 0;
};

} // namespace cuvit
