// Tests the comparison and data-generation helpers the other tests rely on.
// An assertion helper that silently accepts everything would make every kernel
// test pass while proving nothing, so the checker itself is checked here.

#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cuvit::testing::compare_close;
using cuvit::testing::DataGenerator;
using cuvit::testing::require;
using cuvit::testing::TestFailure;

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kQuietNaN = std::numeric_limits<float>::quiet_NaN();

void test_identical_sequences_match() {
    const std::vector<float> values = {0.0F, 1.0F, -2.5F, 1.0e6F, 1.0e-6F};
    const auto result = compare_close(values.data(), values.data(), values.size());
    require(static_cast<bool>(result), "identical sequences did not compare equal");
    require(result.worst_relative_difference == 0.0, "identical sequences reported a difference");
}

void test_single_corrupted_element_is_caught() {
    std::vector<float> expected(64, 1.0F);
    std::vector<float> actual = expected;
    actual[37] = 1.5F;

    const auto result = compare_close(actual.data(), expected.data(), actual.size());
    require(!result, "a corrupted element was accepted");
    require(result.index == 37, "the reported mismatch index is wrong");
    require(result.actual == 1.5 && result.expected == 1.0,
            "the reported values do not name the mismatch");
}

void test_relative_tolerance_scales_with_magnitude() {
    // The same relative error must be judged the same way at any magnitude,
    // which is the property a fixed absolute tolerance lacks.
    const float small_expected = 1.0e-3F;
    const float large_expected = 1.0e6F;
    const float small_actual = small_expected * 1.000001F;
    const float large_actual = large_expected * 1.000001F;

    require(static_cast<bool>(compare_close(&small_actual, &small_expected, 1, 1.0e-5, 0.0)),
            "a 1e-6 relative error was rejected at small magnitude");
    require(static_cast<bool>(compare_close(&large_actual, &large_expected, 1, 1.0e-5, 0.0)),
            "a 1e-6 relative error was rejected at large magnitude");

    // An absolute difference of 1.0 is negligible against 1e6 but total against
    // 1e-3; only the relative term can tell those apart.
    const float shifted_large = large_expected + 1.0F;
    const float shifted_small = small_expected + 1.0F;
    require(static_cast<bool>(compare_close(&shifted_large, &large_expected, 1, 1.0e-5, 0.0)),
            "a negligible error at large magnitude was rejected");
    require(!compare_close(&shifted_small, &small_expected, 1, 1.0e-5, 0.0),
            "a total error at small magnitude was accepted");
}

void test_absolute_tolerance_covers_zero() {
    // Near zero the relative term vanishes, so only atol can admit the rounding
    // noise a reduction leaves behind.
    const float expected = 0.0F;
    const float actual = 1.0e-9F;

    require(!compare_close(&actual, &expected, 1, 1.0e-5, 0.0),
            "a difference from zero was accepted with no absolute tolerance");
    require(static_cast<bool>(compare_close(&actual, &expected, 1, 1.0e-5, 1.0e-6)),
            "rounding noise around zero was rejected");
}

void test_non_finite_values_never_match() {
    const float nan_value = kQuietNaN;
    const float finite = 1.0F;

    auto result = compare_close(&nan_value, &nan_value, 1);
    require(!result, "NaN compared equal to itself");
    require(result.non_finite, "a NaN mismatch was not reported as non-finite");

    require(!compare_close(&nan_value, &finite, 1), "NaN compared equal to a finite value");
    require(!compare_close(&finite, &nan_value, 1), "a finite value compared equal to NaN");

    const float positive_infinity = kInfinity;
    const float negative_infinity = -kInfinity;
    const float huge = 1.0e30F;
    require(static_cast<bool>(compare_close(&positive_infinity, &positive_infinity, 1)),
            "matching infinities were rejected");
    require(!compare_close(&positive_infinity, &negative_infinity, 1),
            "infinities of opposite sign compared equal");
    require(!compare_close(&positive_infinity, &huge, 1),
            "an infinity compared equal to a finite value");
}

void test_worst_difference_is_tracked_on_success() {
    std::vector<float> expected(16, 4.0F);
    std::vector<float> actual = expected;
    actual[9] = 4.0F * (1.0F + 1.0e-6F);

    const auto result = compare_close(actual.data(), expected.data(), actual.size(), 1.0e-5, 0.0);
    require(static_cast<bool>(result), "a within-tolerance difference was rejected");
    require(result.worst_index == 9, "the worst difference was attributed to the wrong index");
    require(result.worst_relative_difference > 0.0,
            "a real difference was reported as no difference");
}

void test_require_close_reports_context() {
    std::vector<float> expected(8, 1.0F);
    std::vector<float> actual = expected;
    actual[3] = 2.0F;

    bool failure_seen = false;
    try {
        cuvit::testing::require_close(actual.data(), expected.data(), actual.size(),
                                      "sample comparison");
    } catch (const TestFailure& error) {
        const std::string message = error.what();
        failure_seen = message.find("sample comparison") != std::string::npos &&
                       message.find("index 3") != std::string::npos;
    }
    require(failure_seen, "require_close did not report the failing context and index");
}

void test_require_exact_rejects_near_but_unequal() {
    const float expected = 1.0F;
    const float actual = std::nextafter(1.0F, 2.0F);

    bool failure_seen = false;
    try {
        cuvit::testing::require_exact(&actual, &expected, 1, "bitwise comparison");
    } catch (const TestFailure&) {
        failure_seen = true;
    }
    require(failure_seen, "require_exact accepted values one ULP apart");

    cuvit::testing::require_exact(&expected, &expected, 1, "identical values");
}

// The property that makes compare_close_scaled the right tool for a reduction:
// cancellation destroys the result-relative error while leaving the
// term-relative error intact.
void test_scaled_comparison_survives_cancellation() {
    // A result left over from nearly cancelling terms: the error is negligible
    // against the terms that were summed, but large against the sum itself.
    const float expected = 1.0e-4F;
    const float actual = 1.1e-4F;
    const float term_scale = 1.0e3F;

    require(!compare_close(&actual, &expected, 1, 1.0e-5, 0.0),
            "result-relative comparison accepted a 10% error");
    require(static_cast<bool>(cuvit::testing::compare_close_scaled(&actual, &expected, &term_scale,
                                                                   1, 1.0e-6, 0.0)),
            "term-relative comparison rejected an error far below one ULP of the terms");
}

void test_scaled_comparison_still_catches_real_errors() {
    // An error that is large against the accumulated terms must still fail,
    // otherwise the scaled form would be a way to make any kernel pass.
    const float expected = 100.0F;
    const float actual = 110.0F;
    const float term_scale = 200.0F;

    const auto result =
        cuvit::testing::compare_close_scaled(&actual, &expected, &term_scale, 1, 1.0e-6, 0.0);
    require(!result, "term-relative comparison accepted a 5% error against the terms");
    require(result.index == 0, "the reported mismatch index is wrong");

    const float nan_scale = kQuietNaN;
    require(!cuvit::testing::compare_close_scaled(&expected, &expected, &nan_scale, 1),
            "a non-finite scale was accepted");
}

void test_generator_is_reproducible() {
    constexpr std::size_t count = 256;
    std::vector<float> first(count);
    std::vector<float> second(count);
    std::vector<float> different(count);

    DataGenerator(1234).normal(first.data(), count);
    DataGenerator(1234).normal(second.data(), count);
    DataGenerator(5678).normal(different.data(), count);

    cuvit::testing::require_exact(first.data(), second.data(), count,
                                  "the same seed produced different data");
    require(first != different, "different seeds produced identical data");
}

void test_generator_respects_bounds() {
    constexpr std::size_t count = 4096;
    std::vector<float> values(count);
    DataGenerator generator(99);
    generator.uniform(values.data(), count, -2.0F, 3.0F);

    for (const float value : values) {
        require(value >= -2.0F && value <= 3.0F, "a uniform sample fell outside its bounds");
        require(std::isfinite(value), "a uniform sample was not finite");
    }
}

} // namespace

CUVIT_TEST_MAIN("numerics_test", {
    test_identical_sequences_match();
    test_single_corrupted_element_is_caught();
    test_relative_tolerance_scales_with_magnitude();
    test_absolute_tolerance_covers_zero();
    test_non_finite_values_never_match();
    test_worst_difference_is_tracked_on_success();
    test_require_close_reports_context();
    test_require_exact_rejects_near_but_unequal();
    test_scaled_comparison_survives_cancellation();
    test_scaled_comparison_still_catches_real_errors();
    test_generator_is_reproducible();
    test_generator_respects_bounds();
})
