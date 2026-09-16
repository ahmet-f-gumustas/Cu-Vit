#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cuvit {

/// How the right-hand operand of a GEMM is stored.
///
/// Both layouts are needed and neither is a transposed copy of the other:
/// torch.nn.Linear keeps its weight as [out_features, in_features] and computes
/// x @ W^T, and attention forms Q @ K^T from two [tokens, head_dim] matrices.
/// Transposing in memory first would cost a full pass over the data for an
/// operation the kernel can express by swapping two index expressions.
enum class GemmLayout {
    /// B is [K, N]: C[m][n] = sum_k A[m][k] * B[k][n].
    kNoTranspose,
    /// B is [N, K]: C[m][n] = sum_k A[m][k] * B[n][k].
    kTransposed,
};

/// What a GEMM does to each result before storing it.
///
/// Folding these into the store saves a second kernel and, more importantly, a
/// full read-modify-write pass over the output. At batch size one the matrices
/// are small enough that a launch and its memory traffic cost as much as the
/// arithmetic, so an epilogue that would be rounding error on a large GEMM is
/// worth having here.
struct GemmEpilogue {
    /// Add into the existing contents of C rather than overwriting them, which
    /// is what a residual connection needs.
    bool accumulate = false;
    /// Apply GELU after the bias, as the first MLP layer does.
    bool gelu = false;
};

/// Batched general matrix multiply with an optional bias and scale.
///
///   C[b][m][n] = alpha * sum_k A[b][m][k] * B_layout[b][k][n] + bias[n]
///
/// Batch elements are addressed by element strides, so a batch can be a slice of
/// a larger tensor: attention runs one GEMM per head over a packed QKV buffer
/// without gathering the heads together first. A batch stride of zero shares one
/// operand across the batch, which is how a single weight matrix is applied to
/// every head.
///
/// @p bias may be null. It is indexed by the output column, matching how a
/// linear layer's bias broadcasts across rows.
/// @p lda, @p ldb and @p ldc give the distance between consecutive rows of each
/// operand, which lets a GEMM read a column slice of a wider matrix in place.
/// Attention relies on this: Q, K and V are interleaved inside one
/// [tokens, 3 * embed] projection, and each head is a further slice of that.
/// Passing zero selects the packed distance for the operand's shape and layout.
void launch_gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
                 GemmLayout layout, float alpha = 1.0F, int batch = 1, std::int64_t stride_a = 0,
                 std::int64_t stride_b = 0, std::int64_t stride_c = 0, int lda = 0, int ldb = 0,
                 int ldc = 0, GemmEpilogue epilogue = {}, cudaStream_t stream = nullptr);

/// Layer normalization over the last axis, with a per-element affine transform.
///
///   out[r][c] = (in[r][c] - mean(in[r])) / sqrt(var(in[r]) + eps) * gamma[c] + beta[c]
///
/// @p rows is the number of independent normalizations and @p columns the length
/// of each. Variance is the biased estimator, matching torch.nn.LayerNorm.
void launch_layer_norm(const float* input, const float* gamma, const float* beta, float* output,
                       int rows, int columns, float epsilon = 1.0e-6F,
                       cudaStream_t stream = nullptr);

/// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
///
/// This is the erf form, not the tanh approximation. timm's ViT uses
/// nn.GELU(approximate='none'). The two forms peak at 4.7e-4 apart around
/// x = -2.7, well above the tolerance a kernel test should accept, so the choice
/// is a correctness matter rather than a performance one.
void launch_gelu(const float* input, float* output, std::int64_t count,
                 cudaStream_t stream = nullptr);

/// Row-wise softmax, computed in the numerically stable form that subtracts the
/// row maximum before exponentiating.
void launch_softmax(const float* input, float* output, int rows, int columns,
                    cudaStream_t stream = nullptr);

/// Rewrites a [channels, height, width] image as a [patches, channels * patch *
/// patch] matrix, one non-overlapping patch per row.
///
/// A ViT patch embedding is a convolution whose kernel equals its stride, so it
/// touches every input element exactly once and reduces to this gather followed
/// by a single GEMM against the flattened convolution weight. Running it as a
/// general convolution would compute the same thing far more slowly.
///
/// Within a row the elements are ordered channel-major, matching how a Conv2d
/// weight of shape [out, in, kh, kw] is laid out.
void launch_extract_patches(const float* image, float* patches, int channels, int height, int width,
                            int patch, cudaStream_t stream = nullptr);

/// Adds @p addend into @p accumulator elementwise, for a residual connection.
void launch_add(float* accumulator, const float* addend, std::int64_t count,
                cudaStream_t stream = nullptr);

} // namespace cuvit
