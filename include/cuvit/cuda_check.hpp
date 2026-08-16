#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cuvit {

class CudaError final : public std::runtime_error {
  public:
    CudaError(cudaError_t code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] cudaError_t code() const noexcept { return code_; }

  private:
    cudaError_t code_;
};

inline void check_cuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line << " ("
            << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << ')';
    throw CudaError(result, message.str());
}

} // namespace cuvit

#define CUVIT_CUDA_CHECK(expression)                                                               \
    ::cuvit::check_cuda((expression), #expression, __FILE__, __LINE__)
