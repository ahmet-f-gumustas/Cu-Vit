#include "cuvit/kernels.hpp"

#include "cuvit/cuda_check.hpp"

#include <cstdint>
#include <stdexcept>

namespace cuvit {
namespace {

// One block computes a 64x64 output tile from 16-deep slices of K. Each of the
// 256 threads owns a 4x4 micro-tile, so every value loaded into shared memory is
// reused four times from registers. A thread-per-output kernel reloads the same
// operands 64 times instead.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadTile = 4;
constexpr int kBlockDim = kTileM / kThreadTile; // 16 threads per side, 256 total

__global__ __launch_bounds__(kBlockDim* kBlockDim) void gemm_kernel(
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ bias,
    float* __restrict__ c, int m, int n, int k, float alpha, bool transposed, std::int64_t stride_a,
    std::int64_t stride_b, std::int64_t stride_c, int lda, int ldb, int ldc) {
    __shared__ float tile_a[kTileK][kTileM];
    __shared__ float tile_b[kTileK][kTileN];

    const std::int64_t batch = blockIdx.z;
    a += batch * stride_a;
    b += batch * stride_b;
    c += batch * stride_c;

    const int row_origin = blockIdx.y * kTileM;
    const int column_origin = blockIdx.x * kTileN;
    const int thread_row = threadIdx.y * kThreadTile;
    const int thread_column = threadIdx.x * kThreadTile;
    const int linear_thread = threadIdx.y * kBlockDim + threadIdx.x;

    float accumulator[kThreadTile][kThreadTile] = {};

    for (int k_origin = 0; k_origin < k; k_origin += kTileK) {
        // 256 threads stage a 64x16 slice of A and a 16x64 slice of B, four
        // elements each. A is transposed into shared memory so the inner loop
        // reads down a column without a stride.
        for (int load = 0; load < kTileM * kTileK / (kBlockDim * kBlockDim); ++load) {
            const int index = linear_thread + load * kBlockDim * kBlockDim;
            const int local_row = index / kTileK;
            const int local_k = index % kTileK;
            const int global_row = row_origin + local_row;
            const int global_k = k_origin + local_k;
            tile_a[local_k][local_row] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::int64_t>(global_row) * lda + global_k]
                    : 0.0F;
        }

        for (int load = 0; load < kTileK * kTileN / (kBlockDim * kBlockDim); ++load) {
            const int index = linear_thread + load * kBlockDim * kBlockDim;
            const int local_k = index / kTileN;
            const int local_column = index % kTileN;
            const int global_k = k_origin + local_k;
            const int global_column = column_origin + local_column;

            float value = 0.0F;
            if (global_k < k && global_column < n) {
                value = transposed ? b[static_cast<std::int64_t>(global_column) * ldb + global_k]
                                   : b[static_cast<std::int64_t>(global_k) * ldb + global_column];
            }
            tile_b[local_k][local_column] = value;
        }

        __syncthreads();

        for (int step = 0; step < kTileK; ++step) {
            float fragment_a[kThreadTile];
            float fragment_b[kThreadTile];
            for (int i = 0; i < kThreadTile; ++i) {
                fragment_a[i] = tile_a[step][thread_row + i];
                fragment_b[i] = tile_b[step][thread_column + i];
            }
            for (int i = 0; i < kThreadTile; ++i) {
                for (int j = 0; j < kThreadTile; ++j) {
                    accumulator[i][j] += fragment_a[i] * fragment_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < kThreadTile; ++i) {
        const int row = row_origin + thread_row + i;
        if (row >= m) {
            continue;
        }
        for (int j = 0; j < kThreadTile; ++j) {
            const int column = column_origin + thread_column + j;
            if (column >= n) {
                continue;
            }
            float value = alpha * accumulator[i][j];
            if (bias != nullptr) {
                value += bias[column];
            }
            c[static_cast<std::int64_t>(row) * ldc + column] = value;
        }
    }
}

} // namespace

void launch_gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
                 GemmLayout layout, float alpha, int batch, std::int64_t stride_a,
                 std::int64_t stride_b, std::int64_t stride_c, int lda, int ldb, int ldc,
                 cudaStream_t stream) {
    if (m == 0 || n == 0 || batch == 0) {
        return;
    }
    if (m < 0 || n < 0 || k < 0 || batch < 0) {
        throw std::invalid_argument("GEMM dimensions cannot be negative");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("GEMM operands cannot be null");
    }

    const bool transposed = layout == GemmLayout::kTransposed;
    if (lda == 0) {
        lda = k;
    }
    if (ldb == 0) {
        ldb = transposed ? k : n;
    }
    if (ldc == 0) {
        ldc = n;
    }
    if (lda < k || ldc < n || ldb < (transposed ? k : n)) {
        throw std::invalid_argument("A GEMM leading dimension is smaller than its row");
    }

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(static_cast<unsigned int>((n + kTileN - 1) / kTileN),
                    static_cast<unsigned int>((m + kTileM - 1) / kTileM),
                    static_cast<unsigned int>(batch));

    gemm_kernel<<<grid, block, 0, stream>>>(a, b, bias, c, m, n, k, alpha, transposed, stride_a,
                                            stride_b, stride_c, lda, ldb, ldc);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuvit
