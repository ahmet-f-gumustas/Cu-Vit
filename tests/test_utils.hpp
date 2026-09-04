#pragma once

#include "cuvit/cuda_check.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvit::testing {

/// Thrown when a check inside a test fails. Distinct from the exceptions the
/// library itself raises, so a test that expects a library error cannot be
/// satisfied by its own assertion machinery.
class TestFailure final : public std::runtime_error {
  public:
    explicit TestFailure(const std::string& message) : std::runtime_error(message) {}
};

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

inline void fail(const std::string& message) { throw TestFailure(message); }

// -- Floating point comparison ------------------------------------------------

/// Where and by how much two sequences disagree.
///
/// A kernel comparison that only answers "equal or not" is not actionable, so a
/// mismatch carries the offending index and both values.
struct Comparison {
    bool matched = true;
    std::size_t index = 0;
    double actual = 0.0;
    double expected = 0.0;
    double absolute_difference = 0.0;
    double relative_difference = 0.0;
    /// Largest relative difference seen, whether or not the comparison passed.
    double worst_relative_difference = 0.0;
    std::size_t worst_index = 0;
    /// Set when a mismatch is a non-finite value rather than a tolerance miss.
    bool non_finite = false;

    [[nodiscard]] explicit operator bool() const noexcept { return matched; }

    [[nodiscard]] std::string describe() const {
        std::ostringstream message;
        if (matched) {
            message << "sequences match (worst relative difference " << worst_relative_difference
                    << " at index " << worst_index << ')';
            return message.str();
        }
        message.precision(9);
        if (non_finite) {
            message << "non-finite value at index " << index << ": actual " << actual
                    << ", expected " << expected;
            return message.str();
        }
        message << "mismatch at index " << index << ": actual " << actual << ", expected "
                << expected << " (absolute " << absolute_difference << ", relative "
                << relative_difference << ')';
        return message.str();
    }
};

/// Compares two sequences with NumPy's mixed tolerance: a value passes when
/// |actual - expected| <= atol + rtol * |expected|.
///
/// A pure absolute tolerance is only meaningful when both sides perform the
/// identical sequence of operations. A GPU reduction accumulates in a different
/// order than a CPU reference, so error grows with the reduction length and must
/// be judged relative to the magnitude of the result.
/// The defaults suit elementwise operations, where each output depends on a
/// handful of inputs. For a reduction, prefer compare_close_scaled: the error of
/// a long accumulation is not proportional to its own result.
template <typename T>
[[nodiscard]] Comparison compare_close(const T* actual, const T* expected, std::size_t count,
                                       double relative_tolerance = 1.0e-5,
                                       double absolute_tolerance = 1.0e-6) {
    static_assert(std::is_floating_point<T>::value, "compare_close needs floating point values");

    Comparison result;
    for (std::size_t index = 0; index < count; ++index) {
        const double lhs = static_cast<double>(actual[index]);
        const double rhs = static_cast<double>(expected[index]);

        if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
            // NaN never compares equal, and an infinity would make the tolerance
            // check meaningless, so both are failures regardless of tolerance.
            if (!(std::isinf(lhs) && std::isinf(rhs) && std::signbit(lhs) == std::signbit(rhs))) {
                result.matched = false;
                result.non_finite = true;
                result.index = index;
                result.actual = lhs;
                result.expected = rhs;
                return result;
            }
            continue;
        }

        const double difference = std::abs(lhs - rhs);
        const double magnitude = std::abs(rhs);
        const double relative = magnitude > 0.0 ? difference / magnitude : difference;

        if (relative > result.worst_relative_difference) {
            result.worst_relative_difference = relative;
            result.worst_index = index;
        }

        if (difference > absolute_tolerance + relative_tolerance * magnitude) {
            result.matched = false;
            result.index = index;
            result.actual = lhs;
            result.expected = rhs;
            result.absolute_difference = difference;
            result.relative_difference = relative;
            return result;
        }
    }
    return result;
}

/// Compares a reduction against a reference using the magnitude of the summed
/// terms as the error scale, rather than the magnitude of the result.
///
/// @p scale must hold, per element, the sum of the absolute values of the terms
/// accumulated into it (for a dot product, sum |a_k * b_k|).
///
/// Judging a reduction by its own result fails under cancellation: when the
/// terms are large and nearly cancel, a rounding error that is negligible next
/// to the terms is enormous next to the sum. Measured over dot products of
/// normally distributed vectors, comparing a 32-wide blocked accumulation
/// against a sequential one, the worst result-relative error swings between
/// 3.8e-5 and 8.2e-4 as K grows from 64 to 3072 and tracks no clean bound, while
/// the term-relative error stays near 2e-7 -- about two ULP -- across that whole
/// range. Only the second is a tolerance a test can be written against.
template <typename T>
[[nodiscard]] Comparison compare_close_scaled(const T* actual, const T* expected, const T* scale,
                                              std::size_t count, double relative_tolerance = 1.0e-6,
                                              double absolute_tolerance = 1.0e-6) {
    static_assert(std::is_floating_point<T>::value,
                  "compare_close_scaled needs floating point values");

    Comparison result;
    for (std::size_t index = 0; index < count; ++index) {
        const double lhs = static_cast<double>(actual[index]);
        const double rhs = static_cast<double>(expected[index]);
        const double magnitude = std::abs(static_cast<double>(scale[index]));

        if (!std::isfinite(lhs) || !std::isfinite(rhs) || !std::isfinite(magnitude)) {
            result.matched = false;
            result.non_finite = true;
            result.index = index;
            result.actual = lhs;
            result.expected = rhs;
            return result;
        }

        const double difference = std::abs(lhs - rhs);
        const double relative = magnitude > 0.0 ? difference / magnitude : difference;

        if (relative > result.worst_relative_difference) {
            result.worst_relative_difference = relative;
            result.worst_index = index;
        }

        if (difference > absolute_tolerance + relative_tolerance * magnitude) {
            result.matched = false;
            result.index = index;
            result.actual = lhs;
            result.expected = rhs;
            result.absolute_difference = difference;
            result.relative_difference = relative;
            return result;
        }
    }
    return result;
}

/// Throws a TestFailure naming the offending element when the sequences differ.
template <typename T>
void require_close(const T* actual, const T* expected, std::size_t count, const char* context,
                   double relative_tolerance = 1.0e-5, double absolute_tolerance = 1.0e-6) {
    const Comparison result =
        compare_close(actual, expected, count, relative_tolerance, absolute_tolerance);
    if (!result) {
        throw TestFailure(std::string(context) + ": " + result.describe());
    }
}

/// Throws a TestFailure when a reduction differs from its reference by more than
/// the tolerance allows, measured against the accumulated term magnitudes.
template <typename T>
void require_close_scaled(const T* actual, const T* expected, const T* scale, std::size_t count,
                          const char* context, double relative_tolerance = 1.0e-6,
                          double absolute_tolerance = 1.0e-6) {
    const Comparison result = compare_close_scaled(actual, expected, scale, count,
                                                   relative_tolerance, absolute_tolerance);
    if (!result) {
        throw TestFailure(std::string(context) + ": " + result.describe());
    }
}

/// Bitwise equality, for paths that must reproduce data exactly rather than
/// approximately: memory copies, fills, and integer kernels.
template <typename T>
void require_exact(const T* actual, const T* expected, std::size_t count, const char* context) {
    for (std::size_t index = 0; index < count; ++index) {
        if (std::memcmp(actual + index, expected + index, sizeof(T)) != 0) {
            std::ostringstream message;
            message << context << ": values differ at index " << index;
            throw TestFailure(message.str());
        }
    }
}

// -- Deterministic data -------------------------------------------------------

/// Fixed-seed generator. Tests must fail reproducibly: a failure that cannot be
/// re-run with the same inputs cannot be debugged.
class DataGenerator final {
  public:
    explicit DataGenerator(std::uint64_t seed = 0x5EEDU) : engine_(seed) {}

    /// Fills with values drawn uniformly from [low, high].
    template <typename T> void uniform(T* data, std::size_t count, T low = -1, T high = 1) {
        std::uniform_real_distribution<T> distribution(low, high);
        for (std::size_t index = 0; index < count; ++index) {
            data[index] = distribution(engine_);
        }
    }

    /// Fills with normally distributed values, the shape neural network weights
    /// and activations actually take.
    template <typename T> void normal(T* data, std::size_t count, T mean = 0, T stddev = 1) {
        std::normal_distribution<T> distribution(mean, stddev);
        for (std::size_t index = 0; index < count; ++index) {
            data[index] = distribution(engine_);
        }
    }

    template <typename T> void integers(T* data, std::size_t count, T low, T high) {
        std::uniform_int_distribution<T> distribution(low, high);
        for (std::size_t index = 0; index < count; ++index) {
            data[index] = distribution(engine_);
        }
    }

  private:
    std::mt19937_64 engine_;
};

// -- Entry point --------------------------------------------------------------

/// Runs @p body, reports the outcome, and maps it to a process exit code.
///
/// Also synchronizes the device before declaring success: a kernel that faults
/// asynchronously would otherwise be reported only by the next CUDA call, or not
/// at all, and the test would pass while the GPU had already failed.
template <typename Body> int run_test(const char* name, Body&& body) {
    try {
        body();
        CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << name << ": PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << name << ": FAIL: " << error.what() << '\n';
        return 1;
    }
}

} // namespace cuvit::testing

/// Declares a test executable's entry point around @p body_statements.
#define CUVIT_TEST_MAIN(test_name, body_statements)                                                \
    int main() { return ::cuvit::testing::run_test(test_name, []() body_statements); }
