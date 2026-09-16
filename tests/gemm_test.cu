#include "cuvit/cuda_check.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/kernels.hpp"
#include "reference/linear_algebra.hpp"
#include "test_utils.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using cuvit::GemmLayout;
using cuvit::testing::DataGenerator;
using cuvit::testing::require;

struct Case {
    int m;
    int n;
    int k;
    const char* label;
};

std::vector<float> run_on_device(const std::vector<float>& a, const std::vector<float>& b,
                                 const std::vector<float>* bias, int m, int n, int k,
                                 GemmLayout layout, float alpha) {
    cuvit::DeviceBuffer<float> device_a(a.size());
    cuvit::DeviceBuffer<float> device_b(b.size());
    cuvit::DeviceBuffer<float> device_c(static_cast<std::size_t>(m) * n);
    cuvit::DeviceBuffer<float> device_bias;

    device_a.copy_from_host(a.data(), a.size());
    device_b.copy_from_host(b.data(), b.size());
    if (bias != nullptr) {
        device_bias.allocate(bias->size());
        device_bias.copy_from_host(bias->data(), bias->size());
    }

    cuvit::GemmProblem problem = cuvit::gemm_problem(m, n, k, layout);
    problem.alpha = alpha;
    cuvit::launch_gemm(device_a.data(), device_b.data(),
                       bias != nullptr ? device_bias.data() : nullptr, device_c.data(), problem);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(static_cast<std::size_t>(m) * n);
    device_c.copy_to_host(output.data(), output.size());
    return output;
}

void check_case(const Case& shape, GemmLayout layout, bool with_bias, float alpha) {
    const std::size_t a_size = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t b_size = static_cast<std::size_t>(shape.k) * shape.n;

    std::vector<float> a(a_size);
    std::vector<float> b(b_size);
    std::vector<float> bias(static_cast<std::size_t>(shape.n));

    DataGenerator generator(static_cast<std::uint64_t>(shape.m * 31 + shape.n * 7 + shape.k));
    generator.normal(a.data(), a.size());
    generator.normal(b.data(), b.size());
    generator.normal(bias.data(), bias.size());

    const bool transposed = layout == GemmLayout::kTransposed;
    std::vector<float> expected(static_cast<std::size_t>(shape.m) * shape.n);
    if (transposed) {
        cuvit::reference::gemm_transposed(a.data(), b.data(), with_bias ? bias.data() : nullptr,
                                          expected.data(), shape.m, shape.n, shape.k, alpha);
    } else {
        cuvit::reference::gemm(a.data(), b.data(), with_bias ? bias.data() : nullptr,
                               expected.data(), shape.m, shape.n, shape.k, alpha);
    }

    const std::vector<float> actual =
        run_on_device(a, b, with_bias ? &bias : nullptr, shape.m, shape.n, shape.k, layout, alpha);

    // The GPU accumulates in a different order than the reference, so the error
    // is judged against the magnitude of the terms summed, not the result. Under
    // cancellation a dot product can be far smaller than the values that formed
    // it, which makes a result-relative tolerance meaningless.
    const std::vector<float> scale = cuvit::reference::gemm_term_magnitudes(
        a.data(), b.data(), shape.m, shape.n, shape.k, transposed, alpha);

    cuvit::testing::require_close_scaled(actual.data(), expected.data(), scale.data(),
                                         actual.size(), shape.label, 1.0e-6, 1.0e-6);
}

void test_shapes_and_layouts() {
    // Every shape the ViT-Tiny forward pass issues, plus the sizes around the
    // 64x64 tile boundary where the bounds checks either work or do not.
    const Case cases[] = {
        {1, 1, 1, "1x1x1"},
        {1, 1000, 192, "classifier head"},
        {17, 33, 7, "small odd"},
        {63, 63, 63, "just under one tile"},
        {64, 64, 64, "exactly one tile"},
        {65, 65, 65, "just over one tile"},
        {197, 576, 192, "qkv projection"},
        {197, 192, 192, "attention output projection"},
        {197, 768, 192, "mlp fc1"},
        {197, 192, 768, "mlp fc2"},
        {197, 197, 64, "attention logits"},
        {197, 64, 197, "attention times values"},
    };

    for (const Case& shape : cases) {
        for (const GemmLayout layout : {GemmLayout::kNoTranspose, GemmLayout::kTransposed}) {
            check_case(shape, layout, false, 1.0F);
            check_case(shape, layout, true, 1.0F);
        }
    }
}

void test_alpha_scales_the_product() {
    // 0.125 is the attention scale for a head dimension of 64.
    check_case({197, 197, 64, "scaled attention logits"}, GemmLayout::kTransposed, false, 0.125F);
    check_case({32, 48, 16, "negative scale"}, GemmLayout::kNoTranspose, true, -2.5F);
}

// The layout attention needs: one GEMM per head over a packed buffer, reached by
// batch strides rather than by gathering the heads together first.
void test_batched_strides() {
    constexpr int heads = 3;
    constexpr int tokens = 197;
    constexpr int head_dim = 64;

    const std::size_t per_head_input = static_cast<std::size_t>(tokens) * head_dim;
    const std::size_t per_head_output = static_cast<std::size_t>(tokens) * tokens;

    std::vector<float> queries(per_head_input * heads);
    std::vector<float> keys(per_head_input * heads);
    DataGenerator generator(2024);
    generator.normal(queries.data(), queries.size());
    generator.normal(keys.data(), keys.size());

    cuvit::DeviceBuffer<float> device_queries(queries.size());
    cuvit::DeviceBuffer<float> device_keys(keys.size());
    cuvit::DeviceBuffer<float> device_output(per_head_output * heads);
    device_queries.copy_from_host(queries.data(), queries.size());
    device_keys.copy_from_host(keys.data(), keys.size());

    cuvit::GemmProblem problem =
        cuvit::gemm_problem(tokens, tokens, head_dim, GemmLayout::kTransposed);
    problem.alpha = 0.125F;
    problem.batch.count = heads;
    problem.batch.stride_a = static_cast<std::int64_t>(per_head_input);
    problem.batch.stride_b = static_cast<std::int64_t>(per_head_input);
    problem.batch.stride_c = static_cast<std::int64_t>(per_head_output);
    cuvit::launch_gemm(device_queries.data(), device_keys.data(), nullptr, device_output.data(),
                       problem);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(per_head_output * heads);
    device_output.copy_to_host(actual.data(), actual.size());

    for (int head = 0; head < heads; ++head) {
        const float* const q = queries.data() + head * per_head_input;
        const float* const k = keys.data() + head * per_head_input;
        std::vector<float> expected(per_head_output);
        cuvit::reference::gemm_transposed(q, k, nullptr, expected.data(), tokens, tokens, head_dim,
                                          0.125F);
        const std::vector<float> scale =
            cuvit::reference::gemm_term_magnitudes(q, k, tokens, tokens, head_dim, true, 0.125F);

        cuvit::testing::require_close_scaled(actual.data() + head * per_head_output,
                                             expected.data(), scale.data(), per_head_output,
                                             "batched attention logits", 1.0e-6, 1.0e-6);
    }
}

// A batch stride of zero shares one operand across the batch, which is how a
// single weight matrix is applied to every head.
void test_zero_stride_shares_an_operand() {
    constexpr int batch = 4;
    constexpr int m = 40;
    constexpr int n = 24;
    constexpr int k = 32;

    std::vector<float> a(static_cast<std::size_t>(batch) * m * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    DataGenerator generator(77);
    generator.normal(a.data(), a.size());
    generator.normal(b.data(), b.size());

    cuvit::DeviceBuffer<float> device_a(a.size());
    cuvit::DeviceBuffer<float> device_b(b.size());
    cuvit::DeviceBuffer<float> device_c(static_cast<std::size_t>(batch) * m * n);
    device_a.copy_from_host(a.data(), a.size());
    device_b.copy_from_host(b.data(), b.size());

    cuvit::GemmProblem problem = cuvit::gemm_problem(m, n, k, GemmLayout::kNoTranspose);
    problem.batch.count = batch;
    problem.batch.stride_a = static_cast<std::int64_t>(m) * k;
    problem.batch.stride_b = 0;
    problem.batch.stride_c = static_cast<std::int64_t>(m) * n;
    cuvit::launch_gemm(device_a.data(), device_b.data(), nullptr, device_c.data(), problem);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(static_cast<std::size_t>(batch) * m * n);
    device_c.copy_to_host(actual.data(), actual.size());

    for (int index = 0; index < batch; ++index) {
        const float* const slice = a.data() + static_cast<std::size_t>(index) * m * k;
        std::vector<float> expected(static_cast<std::size_t>(m) * n);
        cuvit::reference::gemm(slice, b.data(), nullptr, expected.data(), m, n, k);
        const std::vector<float> scale =
            cuvit::reference::gemm_term_magnitudes(slice, b.data(), m, n, k, false);
        cuvit::testing::require_close_scaled(
            actual.data() + static_cast<std::size_t>(index) * m * n, expected.data(), scale.data(),
            expected.size(), "shared weight across a batch", 1.0e-6, 1.0e-6);
    }
}

// The epilogue folds the residual addition and GELU into the store. Both are
// exercised through the model, but only a direct test pins down what they do to
// a single element.
void test_accumulate_epilogue_adds_into_the_destination() {
    constexpr int m = 70;
    constexpr int n = 40;
    constexpr int k = 24;
    const std::size_t count = static_cast<std::size_t>(m) * n;

    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> bias(n);
    std::vector<float> existing(count);
    DataGenerator generator(404);
    generator.normal(a.data(), a.size());
    generator.normal(b.data(), b.size());
    generator.normal(bias.data(), bias.size());
    generator.normal(existing.data(), existing.size());

    std::vector<float> product(count);
    cuvit::reference::gemm(a.data(), b.data(), bias.data(), product.data(), m, n, k);
    std::vector<float> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        expected[index] = existing[index] + product[index];
    }

    cuvit::DeviceBuffer<float> device_a(a.size());
    cuvit::DeviceBuffer<float> device_b(b.size());
    cuvit::DeviceBuffer<float> device_bias(bias.size());
    cuvit::DeviceBuffer<float> device_c(count);
    device_a.copy_from_host(a.data(), a.size());
    device_b.copy_from_host(b.data(), b.size());
    device_bias.copy_from_host(bias.data(), bias.size());
    device_c.copy_from_host(existing.data(), existing.size());

    cuvit::GemmProblem accumulating = cuvit::gemm_problem(m, n, k, GemmLayout::kNoTranspose);
    accumulating.epilogue.accumulate = true;
    cuvit::launch_gemm(device_a.data(), device_b.data(), device_bias.data(), device_c.data(),
                       accumulating);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_c.copy_to_host(actual.data(), count);

    const std::vector<float> scale =
        cuvit::reference::gemm_term_magnitudes(a.data(), b.data(), m, n, k, false);
    cuvit::testing::require_close_scaled(actual.data(), expected.data(), scale.data(), count,
                                         "accumulate epilogue", 1.0e-6, 1.0e-5);

    // Without the epilogue the same call must overwrite instead, or the flag
    // would be doing nothing and the test above would pass on stale data.
    device_c.copy_from_host(existing.data(), existing.size());
    cuvit::launch_gemm(device_a.data(), device_b.data(), device_bias.data(), device_c.data(),
                       cuvit::gemm_problem(m, n, k, GemmLayout::kNoTranspose));
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());
    device_c.copy_to_host(actual.data(), count);
    cuvit::testing::require_close_scaled(actual.data(), product.data(), scale.data(), count,
                                         "default epilogue overwrites", 1.0e-6, 1.0e-5);
}

void test_gelu_epilogue_matches_the_standalone_kernel() {
    constexpr int m = 33;
    constexpr int n = 96;
    constexpr int k = 48;
    const std::size_t count = static_cast<std::size_t>(m) * n;

    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> bias(n);
    DataGenerator generator(505);
    generator.normal(a.data(), a.size());
    generator.normal(b.data(), b.size());
    generator.normal(bias.data(), bias.size());

    std::vector<float> expected(count);
    cuvit::reference::gemm(a.data(), b.data(), bias.data(), expected.data(), m, n, k);
    cuvit::reference::gelu(expected.data(), expected.data(), count);

    cuvit::DeviceBuffer<float> device_a(a.size());
    cuvit::DeviceBuffer<float> device_b(b.size());
    cuvit::DeviceBuffer<float> device_bias(bias.size());
    cuvit::DeviceBuffer<float> device_c(count);
    device_a.copy_from_host(a.data(), a.size());
    device_b.copy_from_host(b.data(), b.size());
    device_bias.copy_from_host(bias.data(), bias.size());

    cuvit::GemmProblem activated = cuvit::gemm_problem(m, n, k, GemmLayout::kNoTranspose);
    activated.epilogue.gelu = true;
    cuvit::launch_gemm(device_a.data(), device_b.data(), device_bias.data(), device_c.data(),
                       activated);
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(count);
    device_c.copy_to_host(actual.data(), count);
    cuvit::testing::require_close(actual.data(), expected.data(), count, "GELU epilogue", 1.0e-4,
                                  1.0e-5);

    // The epilogue must apply GELU after the bias, not before: the two orders
    // differ everywhere the bias is not zero.
    std::vector<float> wrong_order(count);
    cuvit::reference::gemm(a.data(), b.data(), nullptr, wrong_order.data(), m, n, k);
    cuvit::reference::gelu(wrong_order.data(), wrong_order.data(), count);
    for (std::size_t index = 0; index < count; ++index) {
        wrong_order[index] += bias[index % static_cast<std::size_t>(n)];
    }
    require(
        !cuvit::testing::compare_close(actual.data(), wrong_order.data(), count, 1.0e-3, 1.0e-3),
        "the GELU epilogue applied the bias in the wrong order");
}

void test_degenerate_and_invalid_arguments() {
    cuvit::DeviceBuffer<float> buffer(16);

    // An empty problem is a no-op, not an error: the model issues one whenever a
    // dimension collapses.
    cuvit::launch_gemm(buffer.data(), buffer.data(), nullptr, buffer.data(),
                       cuvit::gemm_problem(0, 4, 4, GemmLayout::kNoTranspose));
    cuvit::launch_gemm(buffer.data(), buffer.data(), nullptr, buffer.data(),
                       cuvit::gemm_problem(4, 0, 4, GemmLayout::kNoTranspose));
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    bool caught = false;
    try {
        cuvit::launch_gemm(nullptr, buffer.data(), nullptr, buffer.data(),
                           cuvit::gemm_problem(4, 4, 4, GemmLayout::kNoTranspose));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a null operand was accepted");

    caught = false;
    try {
        cuvit::launch_gemm(buffer.data(), buffer.data(), nullptr, buffer.data(),
                           cuvit::gemm_problem(-1, 4, 4, GemmLayout::kNoTranspose));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a negative dimension was accepted");
}

// K of zero must produce the bias alone, not leave the output untouched.
void test_zero_k_produces_the_bias() {
    constexpr int m = 8;
    constexpr int n = 8;
    std::vector<float> bias(n);
    DataGenerator(5).normal(bias.data(), bias.size());

    cuvit::DeviceBuffer<float> a(1);
    cuvit::DeviceBuffer<float> b(1);
    cuvit::DeviceBuffer<float> device_bias(bias.size());
    cuvit::DeviceBuffer<float> c(static_cast<std::size_t>(m) * n);
    device_bias.copy_from_host(bias.data(), bias.size());

    cuvit::launch_gemm(a.data(), b.data(), device_bias.data(), c.data(),
                       cuvit::gemm_problem(m, n, 0, GemmLayout::kNoTranspose));
    CUVIT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(static_cast<std::size_t>(m) * n);
    c.copy_to_host(actual.data(), actual.size());
    for (int row = 0; row < m; ++row) {
        cuvit::testing::require_exact(actual.data() + static_cast<std::size_t>(row) * n,
                                      bias.data(), n, "a zero-K GEMM did not produce the bias");
    }
}

} // namespace

CUVIT_TEST_MAIN("gemm_test", {
    test_shapes_and_layouts();
    test_alpha_scales_the_product();
    test_batched_strides();
    test_zero_stride_shares_an_operand();
    test_accumulate_epilogue_adds_into_the_destination();
    test_gelu_epilogue_matches_the_standalone_kernel();
    test_degenerate_and_invalid_arguments();
    test_zero_k_produces_the_bias();
})
