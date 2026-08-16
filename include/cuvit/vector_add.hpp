#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuvit {

void launch_vector_add(const float* left, const float* right, float* output, std::size_t count,
                       cudaStream_t stream = nullptr);

} // namespace cuvit
