#include "cuvit/tensor.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using cuvit::Extent;
using cuvit::TensorView;
using cuvit::testing::require;

// Views need no device memory; a host array is enough to check the arithmetic.
std::vector<float> storage(1024, 0.0F);
float* base() { return storage.data(); }

void test_contiguous_layout() {
    const TensorView<float> view(base(), {2, 3, 4});
    require(view.rank() == 3, "rank is wrong");
    require(view.size() == 24, "element count is wrong");
    require(view.bytes() == 24 * sizeof(float), "byte count is wrong");
    require(view.is_contiguous(), "a freshly built view is not contiguous");

    // Row-major: the last axis is the fastest varying.
    require(view.stride(2) == 1, "innermost stride is wrong");
    require(view.stride(1) == 4, "middle stride is wrong");
    require(view.stride(0) == 12, "outermost stride is wrong");

    require(view.offset_of({0, 0, 0}) == 0, "origin is not at offset zero");
    require(view.offset_of({1, 2, 3}) == 23, "offset arithmetic is wrong");
    require(view.shape_string() == "[2, 3, 4]", "shape string is wrong");
}

void test_rank_and_index_validation() {
    const TensorView<float> view(base(), {2, 3});

    bool caught = false;
    try {
        static_cast<void>(view.extent(2));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    require(caught, "an out-of-range axis was accepted");

    caught = false;
    try {
        static_cast<void>(view.offset_of({0}));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "an index of the wrong rank was accepted");

    caught = false;
    try {
        static_cast<void>(view.offset_of({0, 3}));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    require(caught, "an out-of-range index was accepted");

    caught = false;
    try {
        const TensorView<float> too_deep(base(), {1, 1, 1, 1, 1});
        static_cast<void>(too_deep.rank());
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a rank above the maximum was accepted");
}

void test_slice_keeps_rank_and_narrows() {
    const TensorView<float> view(base(), {4, 8});
    const TensorView<float> rows = view.slice(0, 1, 2);

    require(rows.rank() == 2, "slice changed the rank");
    require(rows.extent(0) == 2 && rows.extent(1) == 8, "slice has the wrong shape");
    require(rows.data() == base() + 8, "slice does not start at the requested row");
    require(rows.is_contiguous(), "a full-width row slice is not contiguous");

    // Narrowing the fastest axis leaves gaps, so the result must not claim to be
    // packed: handing it to a routine that assumes packed memory would read the
    // neighbouring rows.
    const TensorView<float> columns = view.slice(1, 2, 3);
    require(columns.extent(1) == 3, "column slice has the wrong width");
    require(columns.stride(0) == 8, "column slice lost the row stride");
    require(!columns.is_contiguous(), "a partial-width slice claimed to be contiguous");

    bool caught = false;
    try {
        static_cast<void>(view.slice(0, 3, 2));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    require(caught, "a slice past the end was accepted");
}

void test_select_drops_an_axis() {
    const TensorView<float> view(base(), {2, 3, 4});
    const TensorView<float> plane = view.select(0, 1);

    require(plane.rank() == 2, "select did not drop an axis");
    require(plane.extent(0) == 3 && plane.extent(1) == 4, "select has the wrong shape");
    require(plane.data() == base() + 12, "select did not advance to the requested entry");
    require(plane.is_contiguous(), "a leading-axis select is not contiguous");

    const TensorView<float> middle = view.select(1, 2);
    require(middle.rank() == 2, "select did not drop the middle axis");
    require(middle.extent(0) == 2 && middle.extent(1) == 4, "middle select has the wrong shape");
    require(middle.stride(0) == 12, "middle select lost the outer stride");
    require(!middle.is_contiguous(), "a middle-axis select claimed to be contiguous");
}

// The layout attention actually needs: read a packed [tokens, heads, head_dim]
// activation as [heads, tokens, head_dim] without moving any data.
void test_transpose_expresses_the_attention_layout() {
    constexpr Extent tokens = 197;
    constexpr Extent heads = 3;
    constexpr Extent head_dim = 64;

    std::vector<float> activation(tokens * heads * head_dim);
    const TensorView<float> packed(activation.data(), {tokens, heads, head_dim});
    const TensorView<float> per_head = packed.transpose(0, 1);

    require(per_head.extent(0) == heads && per_head.extent(1) == tokens,
            "transpose did not exchange the axes");
    require(per_head.data() == packed.data(), "transpose moved the data");
    require(!per_head.is_contiguous(), "the transposed view claimed to be contiguous");

    // Both views must address the same element, which is the whole point of
    // doing this with strides instead of a copy.
    require(packed.offset_of({5, 2, 7}) == per_head.offset_of({2, 5, 7}),
            "the transposed view addresses a different element");

    const TensorView<float> head = per_head.select(0, 1);
    require(head.extent(0) == tokens && head.extent(1) == head_dim,
            "selecting one head has the wrong shape");
    require(head.stride(0) == heads * head_dim, "selecting one head lost the token stride");
}

// Splitting a packed QKV projection into three views, the other layout the
// encoder needs.
void test_qkv_split() {
    constexpr Extent tokens = 197;
    constexpr Extent embed = 192;

    std::vector<float> qkv(tokens * 3 * embed);
    const TensorView<float> packed(qkv.data(), {tokens, 3, embed});

    for (Extent part = 0; part < 3; ++part) {
        const TensorView<float> view = packed.select(1, part);
        require(view.extent(0) == tokens && view.extent(1) == embed,
                "a QKV part has the wrong shape");
        require(view.data() == qkv.data() + part * embed, "a QKV part starts at the wrong offset");
        require(view.stride(0) == 3 * embed, "a QKV part lost the token stride");
        require(!view.is_contiguous(), "an interleaved QKV part claimed to be contiguous");
    }
}

void test_reshape_requires_contiguity() {
    const TensorView<float> view(base(), {6, 4});
    const TensorView<float> flat = view.reshape({24});
    require(flat.rank() == 1 && flat.size() == 24, "reshape produced the wrong shape");
    require(flat.data() == view.data(), "reshape moved the data");

    bool caught = false;
    try {
        static_cast<void>(view.reshape({5, 5}));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a reshape that changes the element count was accepted");

    caught = false;
    try {
        static_cast<void>(view.transpose(0, 1).reshape({24}));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a strided view was reshaped");
}

void test_degenerate_axes_stay_contiguous() {
    // A batch axis of one is the common case for inference; it must not make an
    // otherwise packed tensor look strided.
    const TensorView<float> view(base(), {1, 8, 4});
    require(view.is_contiguous(), "a leading axis of one broke contiguity");
    require(view.transpose(0, 1).is_contiguous(), "transposing a degenerate axis broke contiguity");
}

void test_strided_construction_and_constness() {
    const TensorView<float> view = TensorView<float>::strided(base(), {2, 3}, {16, 2});
    require(view.stride(0) == 16 && view.stride(1) == 2, "explicit strides were not kept");
    require(!view.is_contiguous(), "an explicitly strided view claimed to be contiguous");
    require(view.offset_of({1, 2}) == 20, "strided offset arithmetic is wrong");

    const TensorView<const float> readable = view.as_const();
    require(readable.data() == view.data(), "as_const moved the data");
    require(readable.extent(0) == 2 && readable.stride(0) == 16, "as_const lost the layout");
    static_assert(std::is_const<TensorView<const float>::value_type>::value,
                  "as_const must produce a view of const elements");

    bool caught = false;
    try {
        static_cast<void>(TensorView<float>::strided(base(), {2, 3}, {6}));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a stride list of the wrong length was accepted");
}

} // namespace

CUVIT_TEST_MAIN("tensor_test", {
    test_contiguous_layout();
    test_rank_and_index_validation();
    test_slice_keeps_rank_and_narrows();
    test_select_drops_an_axis();
    test_transpose_expresses_the_attention_layout();
    test_qkv_split();
    test_reshape_requires_contiguity();
    test_degenerate_axes_stay_contiguous();
    test_strided_construction_and_constness();
})
