#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/kernels.hpp"
#include "reference/linear_algebra.hpp"
#include "test_utils.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using cuvit::testing::DataGenerator;
using cuvit::testing::require;

std::vector<float> on_device(const std::vector<float>& input,
                             void (*launch)(const float*, float*, std::int64_t, cudaStream_t)) {
    cuvit::DeviceBuffer<float> in(input.size());
    cuvit::DeviceBuffer<float> out(input.size());
    in.copy_from_host(input.data(), input.size());
    launch(in.data(), out.data(), static_cast<std::int64_t>(input.size()), nullptr);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> output(input.size());
    out.copy_to_host(output.data(), output.size());
    return output;
}

void test_gelu_matches_the_erf_form() {
    constexpr std::size_t count = 4099;
    std::vector<float> input(count);
    DataGenerator(31).normal(input.data(), count, 0.0F, 3.0F);
    // Values where the two GELU formulations differ most, plus the saturating
    // tails.
    const float landmarks[] = {-30.0F, -8.0F, -2.7F, -2.0F, -1.0F, -0.5F, 0.0F,
                               0.5F,   1.0F,  2.0F,  2.7F,  8.0F,  30.0F};
    for (std::size_t index = 0; index < sizeof(landmarks) / sizeof(float); ++index) {
        input[index] = landmarks[index];
    }

    std::vector<float> expected(count);
    cuvit::reference::gelu(input.data(), expected.data(), count);

    const std::vector<float> actual = on_device(input, cuvit::launch_gelu);
    cuvit::testing::require_close(actual.data(), expected.data(), count,
                                  "GELU differs from the CPU reference", 1.0e-5, 1.0e-6);

    // The tanh approximation would pass a loose tolerance while being the wrong
    // function, so check the two are actually distinguishable. The forms are
    // furthest apart near x = -2.7, where they differ by 4.7e-4; at x = 2 the
    // gap is only 9.8e-5 and would slip under this threshold.
    const float probe = -2.7F;
    const float tanh_form =
        0.5F * probe *
        (1.0F + std::tanh(0.7978845608F * (probe + 0.044715F * probe * probe * probe)));
    require(std::abs(tanh_form - cuvit::reference::gelu(probe)) > 4.0e-4F,
            "the test cannot tell the erf and tanh GELU forms apart");
}

void test_residual_add() {
    constexpr std::size_t count = 1024;
    std::vector<float> accumulator(count);
    std::vector<float> addend(count);
    DataGenerator generator(9);
    generator.normal(accumulator.data(), count);
    generator.normal(addend.data(), count);

    std::vector<float> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        expected[index] = accumulator[index] + addend[index];
    }

    cuvit::DeviceBuffer<float> device_accumulator(count);
    cuvit::DeviceBuffer<float> device_addend(count);
    device_accumulator.copy_from_host(accumulator.data(), count);
    device_addend.copy_from_host(addend.data(), count);

    cuvit::launch_add(device_accumulator.data(), device_addend.data(),
                      static_cast<std::int64_t>(count));
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_accumulator.copy_to_host(actual.data(), count);
    // A single addition rounds identically on both sides.
    cuvit::testing::require_exact(actual.data(), expected.data(), count,
                                  "residual add differs from the CPU reference");
}

void test_layer_norm() {
    const int shapes[][2] = {{1, 192}, {197, 192}, {197, 768}, {3, 1}, {5, 31}, {2, 1024}};

    for (const auto& shape : shapes) {
        const int rows = shape[0];
        const int columns = shape[1];
        const std::size_t count = static_cast<std::size_t>(rows) * columns;

        std::vector<float> input(count);
        std::vector<float> gamma(columns);
        std::vector<float> beta(columns);
        DataGenerator generator(static_cast<std::uint64_t>(rows * 101 + columns));
        // Activation statistics a transformer actually produces: the residual
        // stream is roughly zero-mean with unit spread.
        generator.normal(input.data(), count, 0.0F, 1.0F);
        generator.normal(gamma.data(), columns, 1.0F, 0.2F);
        generator.normal(beta.data(), columns, 0.0F, 0.2F);

        std::vector<float> expected(count);
        cuvit::reference::layer_norm(input.data(), gamma.data(), beta.data(), expected.data(), rows,
                                     columns, 1.0e-6F);

        cuvit::DeviceBuffer<float> device_input(count);
        cuvit::DeviceBuffer<float> device_gamma(columns);
        cuvit::DeviceBuffer<float> device_beta(columns);
        cuvit::DeviceBuffer<float> device_output(count);
        device_input.copy_from_host(input.data(), count);
        device_gamma.copy_from_host(gamma.data(), columns);
        device_beta.copy_from_host(beta.data(), columns);

        cuvit::launch_layer_norm(device_input.data(), device_gamma.data(), device_beta.data(),
                                 device_output.data(), rows, columns, 1.0e-6F);
        CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(count);
        device_output.copy_to_host(actual.data(), count);
        cuvit::testing::require_close(actual.data(), expected.data(), count,
                                      "layer norm differs from the CPU reference", 1.0e-5, 2.0e-6);
    }
}

// How far a float implementation can be trusted when a row carries a large
// common offset.
//
// The reference accumulates in double; the kernel works in float, where a value
// near 1e5 is only resolved to 0.0078. Normalization has to recover a deviation
// of order one from numbers of that size, so the accuracy of the result is set
// by the spacing of a float at the input magnitude, not by the algorithm. The
// kernel sums deviations from one element of the row rather than the values
// themselves, which recovers a factor of three; measured worst-case error
// against the double reference:
//
//     offset       direct sum      deviation sum
//          0        4.8e-07            4.8e-07
//         10        1.4e-06            7.2e-07
//       1000        1.0e-04            3.5e-05
//      100000       1.4e-02            4.3e-03
//
// Transformer activations sit at the top of that table, so the tight comparison
// above is the one that matters; this case exists to pin the degradation and
// catch a change that makes it worse.
void test_layer_norm_tolerates_a_large_offset() {
    constexpr int rows = 197;
    constexpr int columns = 192;
    constexpr std::size_t count = static_cast<std::size_t>(rows) * columns;
    constexpr float offset = 1000.0F;

    std::vector<float> input(count);
    const std::vector<float> gamma(columns, 1.0F);
    const std::vector<float> beta(columns, 0.0F);
    DataGenerator(7).normal(input.data(), count, offset, 1.0F);

    std::vector<float> expected(count);
    cuvit::reference::layer_norm(input.data(), gamma.data(), beta.data(), expected.data(), rows,
                                 columns, 1.0e-6F);

    cuvit::DeviceBuffer<float> device_input(count);
    cuvit::DeviceBuffer<float> device_gamma(columns);
    cuvit::DeviceBuffer<float> device_beta(columns);
    cuvit::DeviceBuffer<float> device_output(count);
    device_input.copy_from_host(input.data(), count);
    device_gamma.copy_from_host(gamma.data(), columns);
    device_beta.copy_from_host(beta.data(), columns);

    cuvit::launch_layer_norm(device_input.data(), device_gamma.data(), device_beta.data(),
                             device_output.data(), rows, columns, 1.0e-6F);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_output.copy_to_host(actual.data(), count);

    // Tight enough to fail if the deviation sum is dropped, which measures
    // 1.0e-4 here.
    cuvit::testing::require_close(actual.data(), expected.data(), count,
                                  "layer norm degraded beyond what float allows at this offset",
                                  1.0e-5, 5.0e-5);
}

void test_layer_norm_normalizes() {
    // Independently of the reference: after normalization with unit gamma and
    // zero beta, each row must have zero mean and unit variance.
    constexpr int rows = 8;
    constexpr int columns = 192;
    constexpr std::size_t count = static_cast<std::size_t>(rows) * columns;

    std::vector<float> input(count);
    DataGenerator(3).normal(input.data(), count, 5.0F, 3.0F);
    const std::vector<float> gamma(columns, 1.0F);
    const std::vector<float> beta(columns, 0.0F);

    cuvit::DeviceBuffer<float> device_input(count);
    cuvit::DeviceBuffer<float> device_gamma(columns);
    cuvit::DeviceBuffer<float> device_beta(columns);
    cuvit::DeviceBuffer<float> device_output(count);
    device_input.copy_from_host(input.data(), count);
    device_gamma.copy_from_host(gamma.data(), columns);
    device_beta.copy_from_host(beta.data(), columns);

    cuvit::launch_layer_norm(device_input.data(), device_gamma.data(), device_beta.data(),
                             device_output.data(), rows, columns);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_output.copy_to_host(actual.data(), count);

    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        double sum_squares = 0.0;
        for (int column = 0; column < columns; ++column) {
            const double value = actual[static_cast<std::size_t>(row) * columns + column];
            sum += value;
            sum_squares += value * value;
        }
        require(std::abs(sum / columns) < 1.0e-4, "a normalized row does not have zero mean");
        require(std::abs(sum_squares / columns - 1.0) < 1.0e-3,
                "a normalized row does not have unit variance");
    }
}

void test_softmax() {
    const int shapes[][2] = {{1, 2}, {197, 197}, {4, 1}, {3, 1000}, {8, 33}};

    for (const auto& shape : shapes) {
        const int rows = shape[0];
        const int columns = shape[1];
        const std::size_t count = static_cast<std::size_t>(rows) * columns;

        std::vector<float> input(count);
        DataGenerator(static_cast<std::uint64_t>(rows * 17 + columns))
            .normal(input.data(), count, 0.0F, 4.0F);

        std::vector<float> expected(count);
        cuvit::reference::softmax(input.data(), expected.data(), rows, columns);

        cuvit::DeviceBuffer<float> device_input(count);
        cuvit::DeviceBuffer<float> device_output(count);
        device_input.copy_from_host(input.data(), count);
        cuvit::launch_softmax(device_input.data(), device_output.data(), rows, columns);
        CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(count);
        device_output.copy_to_host(actual.data(), count);
        cuvit::testing::require_close(actual.data(), expected.data(), count,
                                      "softmax differs from the CPU reference", 1.0e-4, 1.0e-6);

        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int column = 0; column < columns; ++column) {
                const float value = actual[static_cast<std::size_t>(row) * columns + column];
                require(value >= 0.0F && value <= 1.0F, "a softmax value is outside [0, 1]");
                sum += value;
            }
            require(std::abs(sum - 1.0) < 1.0e-4, "a softmax row does not sum to one");
        }
    }
}

// Attention logits can be large; the stable form must survive values that would
// overflow expf outright.
void test_softmax_survives_large_logits() {
    constexpr int rows = 4;
    constexpr int columns = 64;
    constexpr std::size_t count = static_cast<std::size_t>(rows) * columns;

    std::vector<float> input(count, 0.0F);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            input[static_cast<std::size_t>(row) * columns + column] =
                column == row ? 500.0F : -500.0F;
        }
    }

    cuvit::DeviceBuffer<float> device_input(count);
    cuvit::DeviceBuffer<float> device_output(count);
    device_input.copy_from_host(input.data(), count);
    cuvit::launch_softmax(device_input.data(), device_output.data(), rows, columns);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_output.copy_to_host(actual.data(), count);

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const float value = actual[static_cast<std::size_t>(row) * columns + column];
            require(std::isfinite(value), "softmax produced a non-finite value on large logits");
            const float expected = column == row ? 1.0F : 0.0F;
            require(std::abs(value - expected) < 1.0e-5F,
                    "softmax of a saturated row is not one-hot");
        }
    }
}

void test_invalid_arguments() {
    cuvit::DeviceBuffer<float> buffer(16);

    cuvit::launch_gelu(buffer.data(), buffer.data(), 0);
    cuvit::launch_softmax(buffer.data(), buffer.data(), 0, 4);
    cuvit::launch_layer_norm(buffer.data(), buffer.data(), buffer.data(), buffer.data(), 0, 4);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    bool caught = false;
    try {
        cuvit::launch_gelu(nullptr, buffer.data(), 4);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "GELU accepted a null operand");

    caught = false;
    try {
        cuvit::launch_layer_norm(buffer.data(), nullptr, buffer.data(), buffer.data(), 2, 4);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "layer norm accepted a null scale");
}

} // namespace

CUVIT_TEST_MAIN("elementwise_test", {
    test_gelu_matches_the_erf_form();
    test_residual_add();
    test_layer_norm();
    test_layer_norm_tolerates_a_large_offset();
    test_layer_norm_normalizes();
    test_softmax();
    test_softmax_survives_large_logits();
    test_invalid_arguments();
})
