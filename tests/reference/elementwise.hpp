#pragma once

#include <cstddef>

/// CPU reference implementations.
///
/// These exist to be obviously correct, not fast. Every kernel is judged against
/// the matching function here, so they are written in the plainest form the
/// operation allows: no blocking, no vectorization, no reordering. Keep them
/// that way even when the optimization is tempting, because a clever reference
/// that shares a mistake with the kernel proves nothing.
namespace cuvit::reference {

inline void vector_add(const float* left, const float* right, float* output, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = left[index] + right[index];
    }
}

} // namespace cuvit::reference
