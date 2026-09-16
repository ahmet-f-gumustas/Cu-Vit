#include "cuvit/kernels.hpp"

#include "cuvit/cuda_check.hpp"

#include <cstdint>
#include <stdexcept>

namespace cuvit {
namespace {

// One block computes a TileM x TileN output tile from TileK-deep slices of K.
// Each thread owns a ThreadTileM x ThreadTileN micro-tile, so every value staged
// in shared memory is reused from registers rather than reloaded.
//
// Two tile shapes are instantiated, because the right one depends on how much
// work there is rather than on the kernel. At batch size one a ViT-Tiny GEMM is
// small -- the most frequent shape is [197, 192] -- and a wide tile covers it in
// a dozen blocks, which measured 0.08 waves per multiprocessor over 36 SMs with
// the SMs at 8% throughput and DRAM at 5%: not enough blocks to occupy the
// machine. A batch stacks the token rows, and the same shapes become tall enough
// that a wider tile's register reuse wins instead. Measured over the five shapes
// a forward pass issues, weighted by how often each runs:
//
//     rows (M)     64x32      128x64
//          197      1660        2554
//          394      2618        2799
//          788      3494        2892
//         1576      5996        4746
//         6304     22647       14305
//
// The crossover sits between 394 and 788 rows, so the dispatch threshold is 512.
template <int TileM, int TileN, int TileK, int ThreadTileM, int ThreadTileN> struct Tile {
    static constexpr int kBlockRows = TileM / ThreadTileM;
    static constexpr int kBlockColumns = TileN / ThreadTileN;
    static constexpr int kThreads = kBlockRows * kBlockColumns;

    static_assert(TileM * TileK % kThreads == 0, "the A tile must divide across the block");
    static_assert(TileK * TileN % kThreads == 0, "the B tile must divide across the block");
};

/// Rows below which the narrow tile wins, from the table above.
constexpr int kLargeTileRows = 512;

template <int TileM, int TileN, int TileK, int ThreadTileM, int ThreadTileN, bool Split>
__global__
__launch_bounds__(Tile<TileM, TileN, TileK, ThreadTileM, ThreadTileN>::kThreads) void gemm_kernel(
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ bias,
    float* __restrict__ c, int m, int n, int k, float alpha, bool transposed, std::int64_t stride_a,
    std::int64_t stride_b, std::int64_t stride_c, int inner, std::int64_t outer_stride_a,
    std::int64_t outer_stride_b, std::int64_t outer_stride_c, int lda, int ldb, int ldc,
    bool accumulate, bool apply_gelu) {
    using Shape = Tile<TileM, TileN, TileK, ThreadTileM, ThreadTileN>;
    constexpr int kThreads = Shape::kThreads;

    __shared__ float tile_a[TileK][TileM];
    __shared__ float tile_b[TileK][TileN];

    // A flat batch advances by one stride; a split one walks an outer and an
    // inner dimension, which is how attention reaches head h of image i inside a
    // single [images, tokens, 3 * embed] projection.
    const int entry = static_cast<int>(blockIdx.z);
    if constexpr (Split) {
        const std::int64_t outer = entry / inner;
        const std::int64_t index = entry % inner;
        a += outer * outer_stride_a + index * stride_a;
        b += outer * outer_stride_b + index * stride_b;
        c += outer * outer_stride_c + index * stride_c;
    } else {
        a += static_cast<std::int64_t>(entry) * stride_a;
        b += static_cast<std::int64_t>(entry) * stride_b;
        c += static_cast<std::int64_t>(entry) * stride_c;
    }

    const int row_origin = blockIdx.y * TileM;
    const int column_origin = blockIdx.x * TileN;
    const int thread_row = threadIdx.y * ThreadTileM;
    const int thread_column = threadIdx.x * ThreadTileN;
    const int linear_thread = threadIdx.y * Shape::kBlockColumns + threadIdx.x;

    float accumulator[ThreadTileM][ThreadTileN] = {};

    for (int k_origin = 0; k_origin < k; k_origin += TileK) {
        // The block stages a TileM x TileK slice of A and a TileK x TileN slice
        // of B. A is transposed into shared memory so the inner loop reads down
        // a column without a stride.
#pragma unroll
        for (int load = 0; load < TileM * TileK / kThreads; ++load) {
            const int index = linear_thread + load * kThreads;
            const int local_row = index / TileK;
            const int local_k = index % TileK;
            const int global_row = row_origin + local_row;
            const int global_k = k_origin + local_k;
            tile_a[local_k][local_row] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::int64_t>(global_row) * lda + global_k]
                    : 0.0F;
        }

#pragma unroll
        for (int load = 0; load < TileK * TileN / kThreads; ++load) {
            const int index = linear_thread + load * kThreads;
            const int local_k = index / TileN;
            const int local_column = index % TileN;
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

#pragma unroll
        for (int step = 0; step < TileK; ++step) {
            float fragment_a[ThreadTileM];
            float fragment_b[ThreadTileN];
#pragma unroll
            for (int i = 0; i < ThreadTileM; ++i) {
                fragment_a[i] = tile_a[step][thread_row + i];
            }
#pragma unroll
            for (int j = 0; j < ThreadTileN; ++j) {
                fragment_b[j] = tile_b[step][thread_column + j];
            }
#pragma unroll
            for (int i = 0; i < ThreadTileM; ++i) {
#pragma unroll
                for (int j = 0; j < ThreadTileN; ++j) {
                    accumulator[i][j] += fragment_a[i] * fragment_b[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < ThreadTileM; ++i) {
        const int row = row_origin + thread_row + i;
        if (row >= m) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < ThreadTileN; ++j) {
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

template <int TileM, int TileN, int TileK, int ThreadTileM, int ThreadTileN>
void launch_tiled(const float* a, const float* b, const float* bias, float* c,
                  const GemmProblem& problem, bool transposed, int lda, int ldb, int ldc,
                  cudaStream_t stream) {
    using Shape = Tile<TileM, TileN, TileK, ThreadTileM, ThreadTileN>;
    const GemmBatch& batch = problem.batch;

    const dim3 block(Shape::kBlockColumns, Shape::kBlockRows);
    const dim3 grid(static_cast<unsigned int>((problem.n + TileN - 1) / TileN),
                    static_cast<unsigned int>((problem.m + TileM - 1) / TileM),
                    static_cast<unsigned int>(batch.count));

    if (batch.inner > 0) {
        gemm_kernel<TileM, TileN, TileK, ThreadTileM, ThreadTileN, true>
            <<<grid, block, 0, stream>>>(a, b, bias, c, problem.m, problem.n, problem.k,
                                         problem.alpha, transposed, batch.stride_a, batch.stride_b,
                                         batch.stride_c, batch.inner, batch.outer_stride_a,
                                         batch.outer_stride_b, batch.outer_stride_c, lda, ldb, ldc,
                                         problem.epilogue.accumulate, problem.epilogue.gelu);
    } else {
        gemm_kernel<TileM, TileN, TileK, ThreadTileM, ThreadTileN, false>
            <<<grid, block, 0, stream>>>(a, b, bias, c, problem.m, problem.n, problem.k,
                                         problem.alpha, transposed, batch.stride_a, batch.stride_b,
                                         batch.stride_c, batch.inner, batch.outer_stride_a,
                                         batch.outer_stride_b, batch.outer_stride_c, lda, ldb, ldc,
                                         problem.epilogue.accumulate, problem.epilogue.gelu);
    }
    CUVIT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_gemm(const float* a, const float* b, const float* bias, float* c,
                 const GemmProblem& problem, cudaStream_t stream) {
    const int m = problem.m;
    const int n = problem.n;
    const int k = problem.k;

    if (m == 0 || n == 0 || problem.batch.count == 0) {
        return;
    }
    if (m < 0 || n < 0 || k < 0 || problem.batch.count < 0) {
        throw std::invalid_argument("GEMM dimensions cannot be negative");
    }
    if (problem.batch.inner > 0 && problem.batch.count % problem.batch.inner != 0) {
        throw std::invalid_argument("A split GEMM batch must divide into whole inner groups");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("GEMM operands cannot be null");
    }

    // One outer group means the split reaches nothing a flat index would not,
    // and the kernel can skip a division by a runtime value at every block.
    GemmProblem flattened = problem;
    if (flattened.batch.inner >= flattened.batch.count) {
        flattened.batch.inner = 0;
    }

    const bool transposed = problem.layout == GemmLayout::kTransposed;
    const int lda = problem.lda != 0 ? problem.lda : k;
    const int ldb = problem.ldb != 0 ? problem.ldb : (transposed ? k : n);
    const int ldc = problem.ldc != 0 ? problem.ldc : n;
    if (lda < k || ldc < n || ldb < (transposed ? k : n)) {
        throw std::invalid_argument("A GEMM leading dimension is smaller than its row");
    }

    // The tile is chosen by how many rows there are, not by the caller: a narrow
    // tile keeps a small problem spread over the SMs, a wide one gives a tall
    // problem more register reuse. Both produce identical results.
    if (m >= kLargeTileRows) {
        launch_tiled<128, 64, 16, 8, 4>(a, b, bias, c, flattened, transposed, lda, ldb, ldc,
                                        stream);
    } else {
        launch_tiled<64, 32, 16, 4, 2>(a, b, bias, c, flattened, transposed, lda, ldb, ldc, stream);
    }
}

} // namespace cuvit
