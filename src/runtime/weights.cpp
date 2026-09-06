#include "cuvit/weights.hpp"

#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace cuvit {
namespace {

/// Layout of a weight file, little-endian throughout:
///
///   magic u32, version u32, entry_count u32, reserved u32
///   entry_count x {
///     name_length u32, name bytes, type u32, rank u32, extents i64[rank],
///     offset u64, bytes u64
///   }
///   payloads, each aligned to 64 bytes
///
/// Offsets are absolute so a reader can seek straight to one tensor without
/// walking the table, and are validated against the file size on load.
constexpr std::uint64_t kPayloadAlignment = 64;

std::uint64_t align_up(std::uint64_t value) {
    return (value + kPayloadAlignment - 1) / kPayloadAlignment * kPayloadAlignment;
}

template <typename T> void write_scalar(std::ostream& stream, T value) {
    static_assert(std::is_trivially_copyable<T>::value, "only trivial values are written");
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T> T read_scalar(const std::vector<std::byte>& blob, std::uint64_t& cursor) {
    if (cursor + sizeof(T) > blob.size()) {
        throw std::runtime_error("Weight file ends inside a header field");
    }
    T value{};
    std::memcpy(&value, blob.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return value;
}

} // namespace

const char* data_type_name(DataType type) {
    switch (type) {
    case DataType::kFloat32:
        return "float32";
    }
    return "unknown";
}

std::size_t data_type_size(DataType type) {
    switch (type) {
    case DataType::kFloat32:
        return sizeof(float);
    }
    throw std::runtime_error("Weight file names an unsupported data type");
}

Extent WeightEntry::element_count() const {
    Extent count = 1;
    for (const Extent extent : extents) {
        count *= extent;
    }
    return count;
}

std::string WeightEntry::shape_string() const {
    std::string text = "[";
    for (std::size_t axis = 0; axis < extents.size(); ++axis) {
        if (axis != 0) {
            text += ", ";
        }
        text += std::to_string(extents[axis]);
    }
    return text + ']';
}

WeightFile WeightFile::load(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }

    const std::streamoff size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine the size of weight file: " + path);
    }
    stream.seekg(0, std::ios::beg);

    WeightFile file;
    file.blob_.resize(static_cast<std::size_t>(size));
    if (!file.blob_.empty() && !stream.read(reinterpret_cast<char*>(file.blob_.data()), size)) {
        throw std::runtime_error("Cannot read weight file: " + path);
    }

    std::uint64_t cursor = 0;
    const auto magic = read_scalar<std::uint32_t>(file.blob_, cursor);
    if (magic != kMagic) {
        throw std::runtime_error("Not a Cu-Vit weight file: " + path);
    }
    const auto version = read_scalar<std::uint32_t>(file.blob_, cursor);
    if (version != kVersion) {
        throw std::runtime_error("Weight file " + path + " has version " + std::to_string(version) +
                                 ", but this build reads version " + std::to_string(kVersion));
    }
    const auto entry_count = read_scalar<std::uint32_t>(file.blob_, cursor);
    static_cast<void>(read_scalar<std::uint32_t>(file.blob_, cursor)); // reserved

    file.entries_.reserve(entry_count);
    for (std::uint32_t index = 0; index < entry_count; ++index) {
        WeightEntry entry;

        const auto name_length = read_scalar<std::uint32_t>(file.blob_, cursor);
        if (cursor + name_length > file.blob_.size()) {
            throw std::runtime_error("Weight file ends inside a tensor name");
        }
        entry.name.assign(reinterpret_cast<const char*>(file.blob_.data() + cursor), name_length);
        cursor += name_length;

        const auto type = read_scalar<std::uint32_t>(file.blob_, cursor);
        if (type != static_cast<std::uint32_t>(DataType::kFloat32)) {
            throw std::runtime_error("Tensor '" + entry.name + "' has unsupported data type " +
                                     std::to_string(type));
        }
        entry.type = static_cast<DataType>(type);

        const auto rank = read_scalar<std::uint32_t>(file.blob_, cursor);
        if (rank == 0 || rank > static_cast<std::uint32_t>(kMaxTensorRank)) {
            throw std::runtime_error("Tensor '" + entry.name + "' has rank " +
                                     std::to_string(rank) + ", outside 1.." +
                                     std::to_string(kMaxTensorRank));
        }
        entry.extents.reserve(rank);
        for (std::uint32_t axis = 0; axis < rank; ++axis) {
            const auto extent = read_scalar<std::int64_t>(file.blob_, cursor);
            if (extent < 0) {
                throw std::runtime_error("Tensor '" + entry.name + "' has a negative extent");
            }
            entry.extents.push_back(extent);
        }

        entry.offset = read_scalar<std::uint64_t>(file.blob_, cursor);
        entry.bytes = read_scalar<std::uint64_t>(file.blob_, cursor);

        const std::uint64_t expected_bytes =
            static_cast<std::uint64_t>(entry.element_count()) * data_type_size(entry.type);
        if (entry.bytes != expected_bytes) {
            throw std::runtime_error("Tensor '" + entry.name + "' declares " +
                                     std::to_string(entry.bytes) + " bytes but its shape " +
                                     entry.shape_string() + " needs " +
                                     std::to_string(expected_bytes));
        }
        if (entry.offset > file.blob_.size() || entry.bytes > file.blob_.size() - entry.offset) {
            throw std::runtime_error("Tensor '" + entry.name + "' lies outside the file");
        }

        if (!file.index_.emplace(entry.name, file.entries_.size()).second) {
            throw std::runtime_error("Weight file contains two tensors named '" + entry.name + "'");
        }
        file.entries_.push_back(std::move(entry));
    }

    return file;
}

bool WeightFile::contains(const std::string& name) const {
    return index_.find(name) != index_.end();
}

const WeightEntry& WeightFile::entry(const std::string& name) const {
    const auto found = index_.find(name);
    if (found == index_.end()) {
        throw std::runtime_error("Weight file has no tensor named '" + name + "'");
    }
    return entries_[found->second];
}

const float* WeightFile::float_data(const std::string& name,
                                    const std::vector<Extent>& expected_extents) const {
    const WeightEntry& found = entry(name);
    if (found.type != DataType::kFloat32) {
        throw std::runtime_error("Tensor '" + name + "' is " + data_type_name(found.type) +
                                 ", not float32");
    }
    if (found.extents != expected_extents) {
        WeightEntry expected;
        expected.extents = expected_extents;
        throw std::runtime_error("Tensor '" + name + "' has shape " + found.shape_string() +
                                 ", but " + expected.shape_string() + " was expected");
    }
    return reinterpret_cast<const float*>(blob_.data() + found.offset);
}

std::uint64_t WeightFile::payload_bytes() const noexcept {
    std::uint64_t total = 0;
    for (const WeightEntry& entry : entries_) {
        total += entry.bytes;
    }
    return total;
}

void WeightFileWriter::add(const std::string& name, const std::vector<Extent>& extents,
                           const float* data) {
    if (name.empty()) {
        throw std::invalid_argument("A weight tensor needs a name");
    }
    if (extents.empty() || extents.size() > static_cast<std::size_t>(kMaxTensorRank)) {
        throw std::invalid_argument("Tensor '" + name + "' has an unsupported rank");
    }

    Extent count = 1;
    for (const Extent extent : extents) {
        if (extent < 0) {
            throw std::invalid_argument("Tensor '" + name + "' has a negative extent");
        }
        count *= extent;
    }

    for (const PendingEntry& existing : entries_) {
        if (existing.name == name) {
            throw std::invalid_argument("Tensor '" + name + "' was added twice");
        }
    }

    PendingEntry entry;
    entry.name = name;
    entry.extents = extents;
    entry.data.assign(data, data + count);
    entries_.push_back(std::move(entry));
}

void WeightFileWriter::write(const std::string& path) const {
    if (entries_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Too many tensors for one weight file");
    }

    // The table is fixed-width per entry apart from the names, so its size is
    // known before anything is written and the payload offsets can be computed
    // up front rather than patched afterwards.
    std::uint64_t table_bytes = 4 * sizeof(std::uint32_t);
    for (const PendingEntry& entry : entries_) {
        table_bytes += sizeof(std::uint32_t) + entry.name.size();
        table_bytes += 2 * sizeof(std::uint32_t);
        table_bytes += entry.extents.size() * sizeof(std::int64_t);
        table_bytes += 2 * sizeof(std::uint64_t);
    }

    std::vector<std::uint64_t> offsets(entries_.size());
    std::uint64_t cursor = align_up(table_bytes);
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        offsets[index] = cursor;
        cursor = align_up(cursor + entries_[index].data.size() * sizeof(float));
    }

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create weight file: " + path);
    }

    write_scalar(stream, WeightFile::kMagic);
    write_scalar(stream, WeightFile::kVersion);
    write_scalar(stream, static_cast<std::uint32_t>(entries_.size()));
    write_scalar(stream, std::uint32_t{0}); // reserved

    for (std::size_t index = 0; index < entries_.size(); ++index) {
        const PendingEntry& entry = entries_[index];
        write_scalar(stream, static_cast<std::uint32_t>(entry.name.size()));
        stream.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
        write_scalar(stream, static_cast<std::uint32_t>(DataType::kFloat32));
        write_scalar(stream, static_cast<std::uint32_t>(entry.extents.size()));
        for (const Extent extent : entry.extents) {
            write_scalar(stream, static_cast<std::int64_t>(extent));
        }
        write_scalar(stream, offsets[index]);
        write_scalar(stream, static_cast<std::uint64_t>(entry.data.size() * sizeof(float)));
    }

    for (std::size_t index = 0; index < entries_.size(); ++index) {
        const std::uint64_t position = static_cast<std::uint64_t>(stream.tellp());
        if (position > offsets[index]) {
            throw std::runtime_error("Weight file layout overran a payload offset");
        }
        const std::vector<char> padding(offsets[index] - position, '\0');
        stream.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        stream.write(reinterpret_cast<const char*>(entries_[index].data.data()),
                     static_cast<std::streamsize>(entries_[index].data.size() * sizeof(float)));
    }

    if (!stream) {
        throw std::runtime_error("Failed while writing weight file: " + path);
    }
}

} // namespace cuvit
