#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

/// CPU reference implementations.
///
/// These exist to be obviously correct, not fast: plain loops, no blocking, no
/// reordering. Keep them that way even when the optimization is tempting,
/// because a clever reference that shares a mistake with the kernel proves
/// nothing.
namespace cuvit::reference {

/// C[m][n] = alpha * sum_k A[m][k] * B[k][n] + bias[n]
inline void gemm(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
                 float alpha = 1.0F) {
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            float sum = 0.0F;
            for (int index = 0; index < k; ++index) {
                sum += a[static_cast<std::int64_t>(row) * k + index] *
                       b[static_cast<std::int64_t>(index) * n + column];
            }
            c[static_cast<std::int64_t>(row) * n + column] =
                alpha * sum + (bias != nullptr ? bias[column] : 0.0F);
        }
    }
}

/// C[m][n] = alpha * sum_k A[m][k] * B[n][k] + bias[n], the layout a linear
/// layer's weight is stored in.
inline void gemm_transposed(const float* a, const float* b, const float* bias, float* c, int m,
                            int n, int k, float alpha = 1.0F) {
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            float sum = 0.0F;
            for (int index = 0; index < k; ++index) {
                sum += a[static_cast<std::int64_t>(row) * k + index] *
                       b[static_cast<std::int64_t>(column) * k + index];
            }
            c[static_cast<std::int64_t>(row) * n + column] =
                alpha * sum + (bias != nullptr ? bias[column] : 0.0F);
        }
    }
}

/// Sum of the absolute products accumulated into each output, the scale a
/// reduction's rounding error should be judged against.
inline std::vector<float> gemm_term_magnitudes(const float* a, const float* b, int m, int n, int k,
                                               bool transposed, float alpha = 1.0F) {
    std::vector<float> scale(static_cast<std::size_t>(m) * n);
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            float total = 0.0F;
            for (int index = 0; index < k; ++index) {
                const float left = a[static_cast<std::int64_t>(row) * k + index];
                const float right = transposed ? b[static_cast<std::int64_t>(column) * k + index]
                                               : b[static_cast<std::int64_t>(index) * n + column];
                total += std::abs(left * right);
            }
            scale[static_cast<std::size_t>(row) * n + column] = std::abs(alpha) * total;
        }
    }
    return scale;
}

inline void layer_norm(const float* input, const float* gamma, const float* beta, float* output,
                       int rows, int columns, float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const float* const source = input + static_cast<std::int64_t>(row) * columns;
        float* const destination = output + static_cast<std::int64_t>(row) * columns;

        double sum = 0.0;
        for (int column = 0; column < columns; ++column) {
            sum += source[column];
        }
        const double mean = sum / columns;

        double variance = 0.0;
        for (int column = 0; column < columns; ++column) {
            const double centered = source[column] - mean;
            variance += centered * centered;
        }
        variance /= columns;

        const double inverse_deviation = 1.0 / std::sqrt(variance + epsilon);
        for (int column = 0; column < columns; ++column) {
            destination[column] =
                static_cast<float>((source[column] - mean) * inverse_deviation) * gamma[column] +
                beta[column];
        }
    }
}

inline void softmax(const float* input, float* output, int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        const float* const source = input + static_cast<std::int64_t>(row) * columns;
        float* const destination = output + static_cast<std::int64_t>(row) * columns;

        float maximum = source[0];
        for (int column = 1; column < columns; ++column) {
            maximum = std::max(maximum, source[column]);
        }

        double sum = 0.0;
        for (int column = 0; column < columns; ++column) {
            const double value = std::exp(static_cast<double>(source[column]) - maximum);
            destination[column] = static_cast<float>(value);
            sum += value;
        }
        for (int column = 0; column < columns; ++column) {
            destination[column] = static_cast<float>(destination[column] / sum);
        }
    }
}

/// The erf form, matching nn.GELU(approximate='none').
inline float gelu(float value) {
    return 0.5F * value * (1.0F + std::erf(value * 0.7071067811865475F));
}

inline void gelu(const float* input, float* output, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = gelu(input[index]);
    }
}

} // namespace cuvit::reference
