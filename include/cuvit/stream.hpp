#pragma once

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <utility>

namespace cuvit {

/// Owns a CUDA stream.
///
/// A default-constructed Stream creates a non-blocking stream, so work queued on
/// it does not implicitly synchronize with the legacy default stream.
///
/// A moved-from Stream is empty and get() then returns nullptr, which CUDA reads
/// as the legacy default stream rather than as an error. Check empty() when the
/// distinction matters.
class Stream final {
  public:
    Stream() : Stream(cudaStreamNonBlocking) {}

    explicit Stream(unsigned int flags) {
        CUVIT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~Stream() { release_noexcept(); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept { swap(other); }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            swap(other);
        }
        return *this;
    }

    /// Blocks the calling thread until every operation queued on the stream has
    /// completed.
    void synchronize() const { CUVIT_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    /// Reports whether all queued work has completed, without blocking.
    [[nodiscard]] bool query() const {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaErrorNotReady) {
            return false;
        }
        CUVIT_CUDA_CHECK(status);
        return true;
    }

    void swap(Stream& other) noexcept {
        using std::swap;
        swap(stream_, other.stream_);
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] bool empty() const noexcept { return stream_ == nullptr; }

  private:
    void release_noexcept() noexcept {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
        stream_ = nullptr;
    }

    cudaStream_t stream_ = nullptr;
};

inline void swap(Stream& left, Stream& right) noexcept { left.swap(right); }

} // namespace cuvit
