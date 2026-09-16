#include "cuvit/kernels.hpp"

#include "cuvit/cuda_check.hpp"

#include <cstdint>
#include <stdexcept>

namespace cuvit {
namespace {

// One block computes a 64x32 output tile from 16-deep slices of K. Each of the
// 256 threads owns a 4x2 micro-tile, so every value staged in shared memory is
// reused from registers rather than reloaded.
//
// The tile is deliberately narrow. At batch size one a ViT-Tiny GEMM is small:
// the largest is [197, 768] and the most frequent is [197, 192], which a 64x64
// tile covers in 12 blocks. Spread over 36 SMs that measured 0.08 waves per
// multiprocessor, with the SMs at 8% throughput and DRAM at 5% -- neither
// compute nor bandwidth bound, simply not enough blocks to occupy the machine.
// Halving the tile width doubles the block count at the same occupancy per
// block, which is worth 1.43x across the shapes this model issues.
//
// Wider tiles win once the matrices are large enough to fill the GPU anyway, so
// this choice belongs with batch-size-one inference, not with the kernel.
constexpr int kTileM = 64;
constexpr int kTileN = 32;
constexpr int kTileK = 16;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileN = 2;
constexpr int kBlockRows = kTileM / kThreadTileM;
constexpr int kBlockColumns = kTileN / kThreadTileN;
constexpr int kThreads = kBlockRows * kBlockColumns;

static_assert(kTileM * kTileK % kThreads == 0, "the A tile must divide across the block");
static_assert(kTileK * kTileN % kThreads == 0, "the B tile must divide across the block");

__global__ __launch_bounds__(kThreads) void gemm_kernel(
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ bias,
    float* __restrict__ c, int m, int n, int k, float alpha, bool transposed, std::int64_t stride_a,
    std::int64_t stride_b, std::int64_t stride_c, int lda, int ldb, int ldc, bool accumulate,
    bool apply_gelu) {
    __shared__ float tile_a[kTileK][kTileM];
    __shared__ float tile_b[kTileK][kTileN];

    const std::int64_t batch = blockIdx.z;
    a += batch * stride_a;
    b += batch * stride_b;
    c += batch * stride_c;

    const int row_origin = blockIdx.y * kTileM;
    const int column_origin = blockIdx.x * kTileN;
    const int thread_row = threadIdx.y * kThreadTileM;
    const int thread_column = threadIdx.x * kThreadTileN;
    const int linear_thread = threadIdx.y * kBlockColumns + threadIdx.x;

    float accumulator[kThreadTileM][kThreadTileN] = {};

    for (int k_origin = 0; k_origin < k; k_origin += kTileK) {
        // 256 threads stage a 64x16 slice of A and a 16x64 slice of B, four
        // elements each. A is transposed into shared memory so the inner loop
        // reads down a column without a stride.
        for (int load = 0; load < kTileM * kTileK / kThreads; ++load) {
            const int index = linear_thread + load * kThreads;
            const int local_row = index / kTileK;
            const int local_k = index % kTileK;
            const int global_row = row_origin + local_row;
            const int global_k = k_origin + local_k;
            tile_a[local_k][local_row] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::int64_t>(global_row) * lda + global_k]
                    : 0.0F;
        }

        for (int load = 0; load < kTileK * kTileN / kThreads; ++load) {
            const int index = linear_thread + load * kThreads;
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
            float fragment_a[kThreadTileM];
            float fragment_b[kThreadTileN];
            for (int i = 0; i < kThreadTileM; ++i) {
                fragment_a[i] = tile_a[step][thread_row + i];
            }
            for (int j = 0; j < kThreadTileN; ++j) {
                fragment_b[j] = tile_b[step][thread_column + j];
            }
            for (int i = 0; i < kThreadTileM; ++i) {
                for (int j = 0; j < kThreadTileN; ++j) {
                    accumulator[i][j] += fragment_a[i] * fragment_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < kThreadTileM; ++i) {
        const int row = row_origin + thread_row + i;
        if (row >= m) {
            continue;
        }
        for (int j = 0; j < kThreadTileN; ++j) {
            const int column = column_origin + thread_column + j;
            if (column >= n) {
                continue;
            }
            float value = alpha * accumulator[i][j];
            if (bias != nullptr) {
                value += bias[column];
            }
            if (apply_gelu) {
                // The erf form, matching nn.GELU(approximate='none').
                value = 0.5F * value * (1.0F + erff(value * 0.7071067811865475F));
            }
            const std::int64_t offset = static_cast<std::int64_t>(row) * ldc + column;
            c[offset] = accumulate ? c[offset] + value : value;
        }
    }
}

} // namespace

void launch_gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
                 GemmLayout layout, float alpha, int batch, std::int64_t stride_a,
                 std::int64_t stride_b, std::int64_t stride_c, int lda, int ldb, int ldc,
                 GemmEpilogue epilogue, cudaStream_t stream) {
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

    const dim3 block(kBlockColumns, kBlockRows);
    const dim3 grid(static_cast<unsigned int>((n + kTileN - 1) / kTileN),
                    static_cast<unsigned int>((m + kTileM - 1) / kTileM),
                    static_cast<unsigned int>(batch));

    gemm_kernel<<<grid, block, 0, stream>>>(a, b, bias, c, m, n, k, alpha, transposed, stride_a,
                                            stride_b, stride_c, lda, ldb, ldc, epilogue.accumulate,
                                            epilogue.gelu);
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuvit
