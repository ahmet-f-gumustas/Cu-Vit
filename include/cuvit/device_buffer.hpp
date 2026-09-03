#pragma once

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvit {

/// Owns a device allocation and the identity of the device holding it.
///
/// Every copy comes in a synchronous form and an `_async` form taking a stream.
/// The asynchronous form only overlaps with compute when the host side is
/// page-locked; see HostBuffer. The host memory and this buffer must both stay
/// alive until the stream has been synchronized.
template <typename T> class DeviceBuffer final {
    static_assert(!std::is_const<T>::value, "DeviceBuffer cannot own const elements");
    static_assert(!std::is_reference<T>::value, "DeviceBuffer cannot own references");
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceBuffer elements must be trivially copyable");

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

    /// Makes the buffer own exactly @p count elements on the active device.
    ///
    /// Requesting the size the buffer already has is a no-op: the existing
    /// allocation, its current contents, and the device it lives on are kept.
    /// Call fill_zero() or reset() first when fresh storage is required.
    void allocate(std::size_t count) {
        if (count == size_) {
            return;
        }

        DeviceBuffer replacement;
        if (count != 0) {
            int device = kNoDevice;
            CUVIT_CUDA_CHECK(cudaGetDevice(&device));

            void* allocation = nullptr;
            CUVIT_CUDA_CHECK(cudaMalloc(&allocation, checked_bytes(count)));
            replacement.data_ = static_cast<T*>(allocation);
            replacement.size_ = count;
            replacement.device_ = device;
        }
        swap(replacement);
    }

    void reset() noexcept { release_noexcept(); }

    void copy_from_host(const T* source, std::size_t count, std::size_t destination_offset = 0) {
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        validate_host_pointer(source, "source");

        CUVIT_CUDA_CHECK(cudaMemcpy(data_ + destination_offset, source, checked_bytes(count),
                                    cudaMemcpyHostToDevice));
    }

    void copy_from_host_async(const T* source, std::size_t count, cudaStream_t stream,
                              std::size_t destination_offset = 0) {
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        validate_host_pointer(source, "source");

        CUVIT_CUDA_CHECK(cudaMemcpyAsync(data_ + destination_offset, source, checked_bytes(count),
                                         cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(T* destination, std::size_t count, std::size_t source_offset = 0) const {
        validate_range(count, source_offset);
        if (count == 0) {
            return;
        }
        validate_host_pointer(destination, "destination");

        CUVIT_CUDA_CHECK(cudaMemcpy(destination, data_ + source_offset, checked_bytes(count),
                                    cudaMemcpyDeviceToHost));
    }

    void copy_to_host_async(T* destination, std::size_t count, cudaStream_t stream,
                            std::size_t source_offset = 0) const {
        validate_range(count, source_offset);
        if (count == 0) {
            return;
        }
        validate_host_pointer(destination, "destination");

        CUVIT_CUDA_CHECK(cudaMemcpyAsync(destination, data_ + source_offset, checked_bytes(count),
                                         cudaMemcpyDeviceToHost, stream));
    }

    void copy_from_device(const DeviceBuffer& source, std::size_t count,
                          std::size_t source_offset = 0, std::size_t destination_offset = 0) {
        source.validate_range(count, source_offset);
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        validate_device_copy(source, count, source_offset, destination_offset);

        CUVIT_CUDA_CHECK(cudaMemcpy(data_ + destination_offset, source.data_ + source_offset,
                                    checked_bytes(count), cudaMemcpyDeviceToDevice));
    }

    void copy_from_device_async(const DeviceBuffer& source, std::size_t count, cudaStream_t stream,
                                std::size_t source_offset = 0, std::size_t destination_offset = 0) {
        source.validate_range(count, source_offset);
        validate_range(count, destination_offset);
        if (count == 0) {
            return;
        }
        validate_device_copy(source, count, source_offset, destination_offset);

        CUVIT_CUDA_CHECK(cudaMemcpyAsync(data_ + destination_offset, source.data_ + source_offset,
                                         checked_bytes(count), cudaMemcpyDeviceToDevice, stream));
    }

    void fill_zero() {
        if (empty()) {
            return;
        }
        CUVIT_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }

    void fill_zero_async(cudaStream_t stream) {
        if (empty()) {
            return;
        }
        CUVIT_CUDA_CHECK(cudaMemsetAsync(data_, 0, bytes(), stream));
    }

    void swap(DeviceBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
        swap(device_, other.device_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Index of the device holding the allocation, or -1 when the buffer is empty.
    [[nodiscard]] int device() const noexcept { return device_; }

  private:
    static constexpr int kNoDevice = -1;

    static std::size_t checked_bytes(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("DeviceBuffer byte size overflow");
        }
        return count * sizeof(T);
    }

    static void validate_host_pointer(const void* pointer, const char* role) {
        if (pointer == nullptr) {
            throw std::invalid_argument(std::string("Host ") + role + " pointer cannot be null");
        }
    }

    void validate_range(std::size_t count, std::size_t offset) const {
        if (offset > size_ || count > size_ - offset) {
            throw std::out_of_range("DeviceBuffer copy exceeds allocation bounds");
        }
    }

    /// Rejects the two device-to-device copies cudaMemcpy cannot express here:
    /// a peer copy across devices, and a self-copy whose ranges overlap.
    /// Ownership is unique, so only a self-copy can ever alias.
    void validate_device_copy(const DeviceBuffer& source, std::size_t count,
                              std::size_t source_offset, std::size_t destination_offset) const {
        if (source.device_ != device_) {
            throw std::invalid_argument(
                "DeviceBuffer copies between different devices are not supported");
        }
        if (this != &source) {
            return;
        }
        const std::size_t distance = source_offset > destination_offset
                                         ? source_offset - destination_offset
                                         : destination_offset - source_offset;
        if (distance < count) {
            throw std::invalid_argument(
                "DeviceBuffer device-to-device copy ranges must not overlap");
        }
    }

    void release_noexcept() noexcept {
        if (data_ != nullptr) {
            // cudaFree needs the owning device current, which is not guaranteed
            // when a buffer outlives a cudaSetDevice elsewhere.
            int previous_device = kNoDevice;
            const bool restore = cudaGetDevice(&previous_device) == cudaSuccess &&
                                 previous_device != device_ &&
                                 cudaSetDevice(device_) == cudaSuccess;
            static_cast<void>(cudaFree(data_));
            if (restore) {
                static_cast<void>(cudaSetDevice(previous_device));
            }
        }
        data_ = nullptr;
        size_ = 0;
        device_ = kNoDevice;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    int device_ = kNoDevice;
};

template <typename T> void swap(DeviceBuffer<T>& left, DeviceBuffer<T>& right) noexcept {
    left.swap(right);
}

} // namespace cuvit
