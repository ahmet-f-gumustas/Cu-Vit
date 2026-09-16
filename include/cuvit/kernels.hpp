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

/// Which batch element a GEMM instance works on.
///
/// The index is optionally split into an outer and an inner part, because
/// attention needs two nested dimensions at once: the projection is
/// [images, tokens, 3 * embed], so one head of one image starts at
/// `image * tokens * 3 * embed + head * head_dim`, and no single linear stride
/// reaches it. Leave @p inner at zero and the index is flat, advancing by the
/// plain strides.
struct GemmBatch {
    /// Number of independent problems.
    int count = 1;
    /// Element distance between consecutive entries.
    std::int64_t stride_a = 0;
    std::int64_t stride_b = 0;
    std::int64_t stride_c = 0;
    /// When positive, entry i is (i / inner, i % inner) and the first part
    /// advances by the outer strides instead.
    int inner = 0;
    std::int64_t outer_stride_a = 0;
    std::int64_t outer_stride_b = 0;
    std::int64_t outer_stride_c = 0;
};

/// A batched general matrix multiply with an optional bias, scale and epilogue.
///
///   C[i][m][n] = epilogue(alpha * sum_k A[i][m][k] * B_layout[i][k][n] + bias[n])
///
/// @p bias may be null. It is indexed by the output column, matching how a
/// linear layer's bias broadcasts across rows.
///
/// @p lda, @p ldb and @p ldc give the distance between consecutive rows of each
/// operand, which lets a GEMM read a column slice of a wider matrix in place.
/// Attention relies on this: Q, K and V are interleaved inside one
/// [tokens, 3 * embed] projection, and each head is a further slice of that.
/// Leaving one at zero selects the packed distance for the operand's shape and
/// layout.
struct GemmProblem {
    int m = 0;
    int n = 0;
    int k = 0;
    GemmLayout layout = GemmLayout::kNoTranspose;
    float alpha = 1.0F;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    GemmBatch batch;
    GemmEpilogue epilogue;
};

/// Builds the common case: one un-batched problem, packed operands, no
/// epilogue. Set the remaining fields on the result for anything else.
[[nodiscard]] inline GemmProblem gemm_problem(int m, int n, int k, GemmLayout layout) {
    GemmProblem problem;
    problem.m = m;
    problem.n = n;
    problem.k = k;
    problem.layout = layout;
    return problem;
}

void launch_gemm(const float* a, const float* b, const float* bias, float* c,
                 const GemmProblem& problem, cudaStream_t stream = nullptr);

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
/// @p images holds @p batch planar images back to back, and @p patches receives
/// their patch rows in the same order.
void launch_extract_patches(const float* images, float* patches, int batch, int channels,
                            int height, int width, int patch, cudaStream_t stream = nullptr);

/// Adds @p addend into @p accumulator elementwise, for a residual connection.
void launch_add(float* accumulator, const float* addend, std::int64_t count,
                cudaStream_t stream = nullptr);

/// Adds one @p addend of @p count elements into each of @p batch consecutive
/// blocks of @p accumulator. The positional embedding is shared by every image
/// in a batch, so it is stored once and broadcast here rather than replicated.
void launch_add_broadcast(float* accumulator, const float* addend, std::int64_t count, int batch,
                          cudaStream_t stream = nullptr);

/// Writes @p source of @p count elements into @p batch slots of @p destination,
/// each @p destination_stride elements apart. Used to seed every image's class
/// token from the single stored one.
void launch_broadcast(float* destination, const float* source, std::int64_t count,
                      std::int64_t destination_stride, int batch, cudaStream_t stream = nullptr);

} // namespace cuvit
