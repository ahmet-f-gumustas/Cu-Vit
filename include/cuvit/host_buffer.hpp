#pragma once

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cuvit {

/// Owns page-locked (pinned) host memory.
///
/// Pinned memory is what makes cudaMemcpyAsync actually asynchronous: a copy
/// issued from pageable memory is staged through a driver bounce buffer and
/// blocks the caller, so DeviceBuffer's async copies only overlap with compute
/// when the host side lives in a HostBuffer.
///
/// Allocation is expensive and pins physical pages for the whole process, so
/// allocate once and reuse rather than creating one per inference step.
template <typename T> class HostBuffer final {
    static_assert(!std::is_const<T>::value, "HostBuffer cannot own const elements");
    static_assert(!std::is_reference<T>::value, "HostBuffer cannot own references");
    static_assert(std::is_trivially_copyable<T>::value,
                  "HostBuffer elements must be trivially copyable");

  public:
    HostBuffer() noexcept = default;

    explicit HostBuffer(std::size_t count) { allocate(count); }

    ~HostBuffer() { release_noexcept(); }

    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    HostBuffer(HostBuffer&& other) noexcept { swap(other); }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            swap(other);
        }
        return *this;
    }

    /// Makes the buffer own exactly @p count pinned elements.
    ///
    /// Requesting the size the buffer already has is a no-op: the existing
    /// allocation and its current contents are kept. Call fill_zero() or
    /// reset() first when fresh storage is required.
    void allocate(std::size_t count) {
        if (count == size_) {
            return;
        }

        HostBuffer replacement;
        if (count != 0) {
            void* allocation = nullptr;
            CUVIT_CUDA_CHECK(cudaMallocHost(&allocation, checked_bytes(count)));
            replacement.data_ = static_cast<T*>(allocation);
            replacement.size_ = count;
        }
        swap(replacement);
    }

    void reset() noexcept { release_noexcept(); }

    void fill_zero() noexcept {
        if (data_ != nullptr) {
            std::memset(data_, 0, bytes());
        }
    }

    void swap(HostBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
    }

    [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    [[nodiscard]] T& at(std::size_t index) {
        if (index >= size_) {
            throw std::out_of_range("HostBuffer index is out of range");
        }
        return data_[index];
    }

    [[nodiscard]] const T& at(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("HostBuffer index is out of range");
        }
        return data_[index];
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] T* begin() noexcept { return data_; }
    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] T* end() noexcept { return data_ + size_; }
    [[nodiscard]] const T* end() const noexcept { return data_ + size_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  private:
    static std::size_t checked_bytes(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("HostBuffer byte size overflow");
        }
        return count * sizeof(T);
    }

    void release_noexcept() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
        }
        data_ = nullptr;
        size_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T> void swap(HostBuffer<T>& left, HostBuffer<T>& right) noexcept {
    left.swap(right);
}

} // namespace cuvit
