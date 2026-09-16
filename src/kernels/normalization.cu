#include "cuvit/kernels.hpp"

#include "cuvit/cuda_check.hpp"

#include <cstdint>
#include <stdexcept>

namespace cuvit {
namespace {

constexpr int kWarpSize = 32;
constexpr int kReduceThreads = 256;

/// Sums @p value across the block, leaving the total in every thread.
__device__ float block_sum(float value, float* scratch) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();

    const int warp_count = blockDim.x / kWarpSize;
    float total = threadIdx.x < warp_count ? scratch[threadIdx.x] : 0.0F;
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
        }
        if (threadIdx.x == 0) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

/// One block per row.
///
/// Mean and variance are taken in two passes over the row rather than from the
/// sum and sum of squares in one. The one-pass form computes var as
/// E[x^2] - E[x]^2, which subtracts two nearly equal numbers when the mean is
/// large relative to the spread and loses most of the significant digits. The
/// rows here are short enough that the second pass costs little.
__global__ __launch_bounds__(kReduceThreads) void layer_norm_kernel(const float* __restrict__ input,
                                                                    const float* __restrict__ gamma,
                                                                    const float* __restrict__ beta,
                                                                    float* __restrict__ output,
                                                                    int columns, float epsilon) {
    __shared__ float scratch[kReduceThreads / kWarpSize];

    const std::int64_t row = blockIdx.x;
    const float* const row_input = input + row * columns;
    float* const row_output = output + row * columns;

    // Summing the row directly loses precision when the values share a large
    // offset: 192 activations around 1000 sum to 192000, where a float's spacing
    // is already 0.015 and swamps the spread the normalization has to resolve.
    // Summing the deviations from one element of the row removes that offset for
    // free, and the mean is recovered exactly by adding it back.
    const float origin = row_input[0];

    float sum = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += row_input[column] - origin;
    }
    const float mean = origin + block_sum(sum, scratch) / static_cast<float>(columns);

    float sum_squares = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float centered = row_input[column] - mean;
        sum_squares += centered * centered;
    }
    const float variance = block_sum(sum_squares, scratch) / static_cast<float>(columns);
    const float inverse_deviation = rsqrtf(variance + epsilon);

    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float normalized = (row_input[column] - mean) * inverse_deviation;
        row_output[column] = normalized * gamma[column] + beta[column];
    }
}

/// One block per row. Softmax is taken in the stable form: subtracting the row
/// maximum before exponentiating leaves the result unchanged mathematically and
/// keeps expf from overflowing on the large logits attention produces.
__global__ __launch_bounds__(kReduceThreads) void softmax_kernel(const float* __restrict__ input,
                                                                 float* __restrict__ output,
                                                                 int columns) {
    __shared__ float scratch[kReduceThreads / kWarpSize];
    __shared__ float shared_maximum;

    const std::int64_t row = blockIdx.x;
    const float* const row_input = input + row * columns;
    float* const row_output = output + row * columns;

    float maximum = -INFINITY;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        maximum = fmaxf(maximum, row_input[column]);
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
    }
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        scratch[warp] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = scratch[0];
        for (int index = 1; index < blockDim.x / kWarpSize; ++index) {
            total = fmaxf(total, scratch[index]);
        }
        shared_maximum = total;
    }
    __syncthreads();
    const float row_maximum = shared_maximum;

    float sum = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = __expf(row_input[column] - row_maximum);
        row_output[column] = value;
        sum += value;
    }
    const float total = block_sum(sum, scratch);
    const float inverse_total = 1.0F / total;

    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        row_output[column] *= inverse_total;
    }
}

constexpr int kElementwiseThreads = 256;

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output,
                            std::int64_t count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        // The erf form, matching nn.GELU(approximate='none').
        output[index] = 0.5F * value * (1.0F + erff(value * 0.7071067811865475F));
    }
}

__global__ void add_kernel(float* __restrict__ accumulator, const float* __restrict__ addend,
                           std::int64_t count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        accumulator[index] += addend[index];
    }
}

/// One thread per output element. The read is scattered and the write is
/// coalesced, which is the right way round: the gather runs once per inference
/// while the matrix it produces is read repeatedly by the GEMM.
__global__ void extract_patches_kernel(const float* __restrict__ image, float* __restrict__ patches,
                                       int channels, int height, int width, int patch,
                                       int patches_across, std::int64_t total) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int patch_area = patch * patch;
    const int row_length = channels * patch_area;

    const int element = static_cast<int>(index % row_length);
    const int patch_index = static_cast<int>(index / row_length);

    const int channel = element / patch_area;
    const int within = element % patch_area;
    const int local_row = within / patch;
    const int local_column = within % patch;

    const int patch_row = patch_index / patches_across;
    const int patch_column = patch_index % patches_across;

    const int source_row = patch_row * patch + local_row;
    const int source_column = patch_column * patch + local_column;

    patches[index] =
        image[(static_cast<std::int64_t>(channel) * height + source_row) * width + source_column];
}

unsigned int elementwise_blocks(std::int64_t count) {
    const std::int64_t blocks = (count + kElementwiseThreads - 1) / kElementwiseThreads;
    if (blocks > 0x7FFFFFFF) {
        throw std::length_error("Elementwise launch exceeds the CUDA grid limit");
    }
    return static_cast<unsigned int>(blocks);
}

} // namespace

void launch_layer_norm(const float* input, const float* gamma, const float* beta, float* output,
                       int rows, int columns, float epsilon, cudaStream_t stream) {
    if (rows == 0 || columns == 0) {
        return;
    }
    if (rows < 0 || columns < 0) {
        throw std::invalid_argument("Layer norm dimensions cannot be negative");
    }
    if (input == nullptr || gamma == nullptr || beta == nullptr || output == nullptr) {
        throw std::invalid_argument("Layer norm operands cannot be null");
    }

    layer_norm_kernel<<<static_cast<unsigned int>(rows), kReduceThreads, 0, stream>>>(
        input, gamma, beta, output, columns, epsilon);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

void launch_softmax(const float* input, float* output, int rows, int columns, cudaStream_t stream) {
    if (rows == 0 || columns == 0) {
        return;
    }
    if (rows < 0 || columns < 0) {
        throw std::invalid_argument("Softmax dimensions cannot be negative");
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("Softmax operands cannot be null");
    }

    softmax_kernel<<<static_cast<unsigned int>(rows), kReduceThreads, 0, stream>>>(input, output,
                                                                                   columns);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

void launch_gelu(const float* input, float* output, std::int64_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    if (count < 0) {
        throw std::invalid_argument("GELU element count cannot be negative");
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("GELU operands cannot be null");
    }

    gelu_kernel<<<elementwise_blocks(count), kElementwiseThreads, 0, stream>>>(input, output,
                                                                               count);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

void launch_extract_patches(const float* image, float* patches, int channels, int height, int width,
                            int patch, cudaStream_t stream) {
    if (channels <= 0 || height <= 0 || width <= 0 || patch <= 0) {
        throw std::invalid_argument("Patch extraction dimensions must be positive");
    }
    if (height % patch != 0 || width % patch != 0) {
        throw std::invalid_argument("Patch extraction needs the image to divide into whole "
                                    "patches");
    }
    if (image == nullptr || patches == nullptr) {
        throw std::invalid_argument("Patch extraction operands cannot be null");
    }

    const int patches_down = height / patch;
    const int patches_across = width / patch;
    const std::int64_t total =
        static_cast<std::int64_t>(patches_down) * patches_across * channels * patch * patch;

    extract_patches_kernel<<<elementwise_blocks(total), kElementwiseThreads, 0, stream>>>(
        image, patches, channels, height, width, patch, patches_across, total);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

void launch_add(float* accumulator, const float* addend, std::int64_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    if (count < 0) {
        throw std::invalid_argument("Residual add element count cannot be negative");
    }
    if (accumulator == nullptr || addend == nullptr) {
        throw std::invalid_argument("Residual add operands cannot be null");
    }

    add_kernel<<<elementwise_blocks(count), kElementwiseThreads, 0, stream>>>(accumulator, addend,
                                                                              count);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuvit
