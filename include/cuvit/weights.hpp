#pragma once

#include "cuvit/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuvit {

/// Element type of a stored tensor. Only the types the engine actually reads are
/// listed; an unknown value in a file is an error rather than something to skip,
/// so a weight the engine cannot interpret is never silently ignored.
enum class DataType : std::uint32_t {
    kFloat32 = 1,
};

[[nodiscard]] const char* data_type_name(DataType type);
[[nodiscard]] std::size_t data_type_size(DataType type);

/// One tensor inside a weight file.
struct WeightEntry {
    std::string name;
    DataType type = DataType::kFloat32;
    std::vector<Extent> extents;
    /// Offset of the payload from the start of the file.
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;

    [[nodiscard]] Extent element_count() const;
    [[nodiscard]] std::string shape_string() const;
};

/// A weight file read into host memory.
///
/// The format is deliberately flat and versioned: a header, a table of entries,
/// then the payloads. Reading it needs no third-party parser, and a file written
/// by an older exporter is rejected by version rather than misread.
///
/// Lookup is by name, and a missing or mis-shaped tensor is an error at load
/// time. A model that silently ran with an uninitialized weight would produce
/// plausible-looking output and waste far more time than a failed load.
class WeightFile final {
  public:
    static constexpr std::uint32_t kMagic = 0x54495643U; // "CVIT"
    static constexpr std::uint32_t kVersion = 1;

    /// Reads @p path, validating the header, the entry table, and that every
    /// payload lies inside the file.
    static WeightFile load(const std::string& path);

    [[nodiscard]] bool contains(const std::string& name) const;

    /// Returns the entry for @p name, or throws naming what was missing.
    [[nodiscard]] const WeightEntry& entry(const std::string& name) const;

    /// Returns a host pointer to the payload of @p name after checking that its
    /// type and shape match what the caller expects.
    [[nodiscard]] const float* float_data(const std::string& name,
                                          const std::vector<Extent>& expected_extents) const;

    [[nodiscard]] const std::vector<WeightEntry>& entries() const noexcept { return entries_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    /// Total bytes every payload occupies, useful for sizing an arena.
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept;

  private:
    std::vector<WeightEntry> entries_;
    std::unordered_map<std::string, std::size_t> index_;
    std::vector<std::byte> blob_;
};

/// Writes a weight file. Used by the tests and by tooling that produces one
/// without going through the Python exporter.
class WeightFileWriter final {
  public:
    void add(const std::string& name, const std::vector<Extent>& extents, const float* data);
    void write(const std::string& path) const;
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  private:
    struct PendingEntry {
        std::string name;
        std::vector<Extent> extents;
        std::vector<float> data;
    };

    std::vector<PendingEntry> entries_;
};

} // namespace cuvit
