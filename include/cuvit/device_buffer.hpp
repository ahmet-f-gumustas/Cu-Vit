#pragma once

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cuvit {

template <typename T> class DeviceBuffer final {
    static_assert(!std::is_const<T>::value, "DeviceBuffer cannot own const elements");
    static_assert(!std::is_reference<T>::value, "DeviceBuffer cannot own references");

  public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release_noexcept(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            swap(other);
        }
        return *this;
    }

    /// Makes the buffer own exactly @p count elements.
    ///
    /// Requesting the size the buffer already has is a no-op: the existing
    /// allocation and its current contents are kept. Call fill_zero() or
    /// reset() first when fresh storage is required.
    void allocate(std::size_t count) {
        if (count == size_) {
            return;
        }

        DeviceBuffer replacement;
        if (count != 0) {
            void* allocation = nullptr;
            CUVIT_CUDA_CHECK(cudaMalloc(&allocation, checked_bytes(count)));
            replacement.data_ = static_cast<T*>(allocation);
            replacement.size_ = count;
        }
        swap(replacement);
    }

    void reset() noexcept { release_noexcept(); }

    void copy_from_host(const T* source, std::size_t count, std::size_t destination_offset = 0) {
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        if (source == nullptr) {
            throw std::invalid_argument("Host source pointer cannot be null");
        }

        CUVIT_CUDA_CHECK(cudaMemcpy(data_ + destination_offset, source, checked_bytes(count),
                                    cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* destination, std::size_t count, std::size_t source_offset = 0) const {
        validate_range(count, source_offset);
        if (count == 0) {
            return;
        }
        if (destination == nullptr) {
            throw std::invalid_argument("Host destination pointer cannot be null");
        }

        CUVIT_CUDA_CHECK(cudaMemcpy(destination, data_ + source_offset, checked_bytes(count),
                                    cudaMemcpyDeviceToHost));
    }

    void copy_from_device(const DeviceBuffer& source, std::size_t count,
                          std::size_t source_offset = 0, std::size_t destination_offset = 0) {
        source.validate_range(count, source_offset);
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        if (this == &source) {
            const std::size_t distance = source_offset > destination_offset
                                             ? source_offset - destination_offset
                                             : destination_offset - source_offset;
            if (distance < count) {
                throw std::invalid_argument(
                    "DeviceBuffer device-to-device copy ranges must not overlap");
            }
        }

        CUVIT_CUDA_CHECK(cudaMemcpy(data_ + destination_offset, source.data_ + source_offset,
                                    checked_bytes(count), cudaMemcpyDeviceToDevice));
    }

    void fill_zero() {
        if (empty()) {
            return;
        }
        CUVIT_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }

    void swap(DeviceBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  private:
    static std::size_t checked_bytes(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("DeviceBuffer byte size overflow");
        }
        return count * sizeof(T);
    }

    void validate_range(std::size_t count, std::size_t offset) const {
        if (offset > size_ || count > size_ - offset) {
            throw std::out_of_range("DeviceBuffer copy exceeds allocation bounds");
        }
    }

    void release_noexcept() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
        data_ = nullptr;
        size_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T> void swap(DeviceBuffer<T>& left, DeviceBuffer<T>& right) noexcept {
    left.swap(right);
}

} // namespace cuvit
