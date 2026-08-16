#include "cuvit/vector_add.hpp"

#include "cuvit/cuda_check.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cuvit {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;

__global__ void vector_add_kernel(const float* left, const float* right, float* output,
                                  std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

} // namespace

void launch_vector_add(const float* left, const float* right, float* output, std::size_t count,
                       cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    if (left == nullptr || right == nullptr || output == nullptr) {
        throw std::invalid_argument("Vector-add device pointers cannot be null");
    }

    const std::size_t block_count = 1 + (count - 1) / kThreadsPerBlock;
    if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Vector-add launch exceeds the CUDA grid limit");
    }

    vector_add_kernel<<<static_cast<unsigned int>(block_count), kThreadsPerBlock, 0, stream>>>(
        left, right, output, count);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuvit
