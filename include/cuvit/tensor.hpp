#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvit {

/// Largest rank the engine represents. A vision transformer never exceeds
/// [batch, heads, tokens, head_dim].
inline constexpr int kMaxTensorRank = 4;

using Extent = std::int64_t;

/// A non-owning, strided view of a tensor.
///
/// Ownership stays with DeviceBuffer or DeviceArena; a view is cheap to copy and
/// carries no lifetime of its own, so slicing and reshaping never raise a
/// question about who frees the memory. The caller must keep the underlying
/// allocation alive for as long as any view of it is used.
///
/// Strides are explicit rather than implied by the shape, because attention
/// needs to read one buffer through several layouts: splitting a packed QKV
/// projection into three views, and reading [batch, tokens, heads, head_dim] as
/// [batch, heads, tokens, head_dim]. Both are stride changes over memory that
/// must not be copied.
///
/// Strides are counted in elements, not bytes.
template <typename T> class TensorView {
    static_assert(!std::is_reference<T>::value, "TensorView cannot view references");

  public:
    using value_type = T;

    TensorView() noexcept = default;

    /// Views @p data as a contiguous tensor in row-major order, where the last
    /// axis is the fastest varying.
    TensorView(T* data, std::initializer_list<Extent> extents) {
        assign_extents(extents);
        fill_contiguous_strides();
        data_ = data;
        validate_data();
    }

    /// Views @p data with explicit strides, for a layout that is not contiguous.
    static TensorView strided(T* data, std::initializer_list<Extent> extents,
                              std::initializer_list<Extent> strides) {
        if (extents.size() != strides.size()) {
            throw std::invalid_argument("TensorView needs one stride per extent");
        }
        TensorView view;
        view.assign_extents(extents);
        int axis = 0;
        for (const Extent stride : strides) {
            if (stride < 0) {
                throw std::invalid_argument("TensorView strides cannot be negative");
            }
            view.strides_[axis++] = stride;
        }
        view.data_ = data;
        view.validate_data();
        return view;
    }

    /// Reinterprets a view of mutable elements as a view of const elements.
    [[nodiscard]] TensorView<const T> as_const() const noexcept {
        TensorView<const T> view;
        view.adopt(data_, extents_, strides_, rank_);
        return view;
    }

    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] T* data() const noexcept { return data_; }

    [[nodiscard]] Extent extent(int axis) const { return extents_[checked_axis(axis)]; }
    [[nodiscard]] Extent stride(int axis) const { return strides_[checked_axis(axis)]; }

    /// Number of elements the view addresses, which is not the size of the
    /// underlying allocation when the view is strided.
    [[nodiscard]] Extent size() const noexcept {
        Extent total = 1;
        for (int axis = 0; axis < rank_; ++axis) {
            total *= extents_[axis];
        }
        return total;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return static_cast<std::size_t>(size()) * sizeof(T);
    }

    /// True when the elements are laid out densely in row-major order, so the
    /// view can be handed to a routine that assumes packed memory.
    [[nodiscard]] bool is_contiguous() const noexcept {
        Extent expected = 1;
        for (int axis = rank_ - 1; axis >= 0; --axis) {
            // A degenerate axis imposes no constraint: nothing is addressed
            // along it, so any stride describes the same empty set.
            if (extents_[axis] == 1) {
                continue;
            }
            if (strides_[axis] != expected) {
                return false;
            }
            expected *= extents_[axis];
        }
        return true;
    }

    /// Element offset of @p indices from the start of the view.
    [[nodiscard]] Extent offset_of(std::initializer_list<Extent> indices) const {
        if (static_cast<int>(indices.size()) != rank_) {
            throw std::invalid_argument("TensorView index has the wrong rank");
        }
        Extent offset = 0;
        int axis = 0;
        for (const Extent index : indices) {
            if (index < 0 || index >= extents_[axis]) {
                throw std::out_of_range("TensorView index is out of range on axis " +
                                        std::to_string(axis));
            }
            offset += index * strides_[axis];
            ++axis;
        }
        return offset;
    }

    /// Narrows @p axis to @p count entries starting at @p start, keeping the rank.
    [[nodiscard]] TensorView slice(int axis, Extent start, Extent count) const {
        const int checked = checked_axis(axis);
        if (start < 0 || count < 0 || start > extents_[checked] ||
            count > extents_[checked] - start) {
            throw std::out_of_range("TensorView slice exceeds the extent of axis " +
                                    std::to_string(axis));
        }
        TensorView view = *this;
        view.data_ = data_ + start * strides_[checked];
        view.extents_[checked] = count;
        return view;
    }

    /// Selects one entry along @p axis and drops that axis, lowering the rank.
    [[nodiscard]] TensorView select(int axis, Extent index) const {
        const int checked = checked_axis(axis);
        if (index < 0 || index >= extents_[checked]) {
            throw std::out_of_range("TensorView select is out of range on axis " +
                                    std::to_string(axis));
        }
        TensorView view;
        view.data_ = data_ + index * strides_[checked];
        view.rank_ = rank_ - 1;
        int target = 0;
        for (int source = 0; source < rank_; ++source) {
            if (source == checked) {
                continue;
            }
            view.extents_[target] = extents_[source];
            view.strides_[target] = strides_[source];
            ++target;
        }
        return view;
    }

    /// Exchanges two axes without moving any data.
    [[nodiscard]] TensorView transpose(int first, int second) const {
        const int left = checked_axis(first);
        const int right = checked_axis(second);
        TensorView view = *this;
        std::swap(view.extents_[left], view.extents_[right]);
        std::swap(view.strides_[left], view.strides_[right]);
        return view;
    }

    /// Reinterprets the shape, keeping the element order.
    ///
    /// Only a contiguous view can be reshaped: over a strided view the operation
    /// is not expressible without moving data, and silently producing a wrong
    /// layout is worse than refusing.
    [[nodiscard]] TensorView reshape(std::initializer_list<Extent> extents) const {
        if (!is_contiguous()) {
            throw std::invalid_argument("Only a contiguous TensorView can be reshaped");
        }
        TensorView view;
        view.assign_extents(extents);
        if (view.size() != size()) {
            throw std::invalid_argument("TensorView reshape changes the element count");
        }
        view.fill_contiguous_strides();
        view.data_ = data_;
        return view;
    }

    [[nodiscard]] std::string shape_string() const {
        std::string text = "[";
        for (int axis = 0; axis < rank_; ++axis) {
            if (axis != 0) {
                text += ", ";
            }
            text += std::to_string(extents_[axis]);
        }
        return text + ']';
    }

  private:
    template <typename U> friend class TensorView;

    void adopt(T* data, const std::array<Extent, kMaxTensorRank>& extents,
               const std::array<Extent, kMaxTensorRank>& strides, int rank) noexcept {
        data_ = data;
        extents_ = extents;
        strides_ = strides;
        rank_ = rank;
    }

    void assign_extents(std::initializer_list<Extent> extents) {
        if (extents.size() == 0 || static_cast<int>(extents.size()) > kMaxTensorRank) {
            throw std::invalid_argument("TensorView rank must be between 1 and " +
                                        std::to_string(kMaxTensorRank));
        }
        rank_ = static_cast<int>(extents.size());
        int axis = 0;
        for (const Extent extent : extents) {
            if (extent < 0) {
                throw std::invalid_argument("TensorView extents cannot be negative");
            }
            extents_[axis++] = extent;
        }
    }

    void fill_contiguous_strides() noexcept {
        Extent stride = 1;
        for (int axis = rank_ - 1; axis >= 0; --axis) {
            strides_[axis] = stride;
            stride *= extents_[axis];
        }
    }

    void validate_data() const {
        if (data_ == nullptr && size() != 0) {
            throw std::invalid_argument("A non-empty TensorView cannot view a null pointer");
        }
    }

    [[nodiscard]] int checked_axis(int axis) const {
        if (axis < 0 || axis >= rank_) {
            throw std::out_of_range("TensorView axis " + std::to_string(axis) +
                                    " is outside a rank-" + std::to_string(rank_) + " tensor");
        }
        return axis;
    }

    T* data_ = nullptr;
    std::array<Extent, kMaxTensorRank> extents_{};
    std::array<Extent, kMaxTensorRank> strides_{};
    int rank_ = 0;
};

} // namespace cuvit
