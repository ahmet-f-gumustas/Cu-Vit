#include "cuvit/device_arena.hpp"
#include "cuvit/tensor.hpp"
#include "cuvit/weights.hpp"
#include "test_utils.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuvit::Extent;
using cuvit::WeightFile;
using cuvit::WeightFileWriter;
using cuvit::testing::DataGenerator;
using cuvit::testing::require;

/// Removes its file when it goes out of scope, so a failing assertion does not
/// leave test artifacts behind.
class ScopedFile final {
  public:
    explicit ScopedFile(std::string path) : path_(std::move(path)) {}
    ~ScopedFile() { std::remove(path_.c_str()); }
    ScopedFile(const ScopedFile&) = delete;
    ScopedFile& operator=(const ScopedFile&) = delete;

    [[nodiscard]] const std::string& path() const noexcept { return path_; }

  private:
    std::string path_;
};

std::string temporary_path(const char* name) { return std::string("cuvit_test_") + name; }

void test_round_trip_preserves_shape_and_data() {
    const ScopedFile file(temporary_path("round_trip.cvw"));

    // Shapes taken from ViT-Tiny/16: a rank-4 patch embedding kernel, a rank-2
    // projection, and a rank-1 bias.
    std::vector<float> patch(192 * 3 * 16 * 16);
    std::vector<float> projection(192 * 192);
    std::vector<float> bias(192);

    DataGenerator generator(4242);
    generator.normal(patch.data(), patch.size());
    generator.normal(projection.data(), projection.size());
    generator.normal(bias.data(), bias.size());

    WeightFileWriter writer;
    writer.add("patch_embed.weight", {192, 3, 16, 16}, patch.data());
    writer.add("blocks.0.attn.proj.weight", {192, 192}, projection.data());
    writer.add("blocks.0.attn.proj.bias", {192}, bias.data());
    writer.write(file.path());

    const WeightFile loaded = WeightFile::load(file.path());
    require(loaded.size() == 3, "the loaded file has the wrong tensor count");
    require(loaded.contains("patch_embed.weight"), "a written tensor is missing");
    require(!loaded.contains("absent"), "an absent tensor was reported present");

    const cuvit::WeightEntry& entry = loaded.entry("patch_embed.weight");
    require(entry.extents == std::vector<Extent>({192, 3, 16, 16}), "the shape was not preserved");
    require(entry.element_count() == 192 * 3 * 16 * 16, "the element count is wrong");
    require(entry.bytes == entry.element_count() * sizeof(float), "the byte count is wrong");

    cuvit::testing::require_exact(loaded.float_data("patch_embed.weight", {192, 3, 16, 16}),
                                  patch.data(), patch.size(),
                                  "the patch embedding did not survive the round trip");
    cuvit::testing::require_exact(loaded.float_data("blocks.0.attn.proj.weight", {192, 192}),
                                  projection.data(), projection.size(),
                                  "the projection did not survive the round trip");
    cuvit::testing::require_exact(loaded.float_data("blocks.0.attn.proj.bias", {192}), bias.data(),
                                  bias.size(), "the bias did not survive the round trip");
}

void test_payloads_are_aligned() {
    const ScopedFile file(temporary_path("alignment.cvw"));

    // Odd sizes, so a writer that did not pad would leave the later payloads at
    // arbitrary offsets.
    std::vector<float> values(7, 1.0F);
    WeightFileWriter writer;
    writer.add("a", {7}, values.data());
    writer.add("b", {7}, values.data());
    writer.add("c", {7}, values.data());
    writer.write(file.path());

    const WeightFile loaded = WeightFile::load(file.path());
    for (const cuvit::WeightEntry& entry : loaded.entries()) {
        require(entry.offset % alignof(float) == 0,
                "a payload is not aligned for the type it holds");
        require(entry.offset % 64 == 0, "a payload does not sit on the declared alignment");
    }
}

void test_shape_mismatch_is_rejected() {
    const ScopedFile file(temporary_path("mismatch.cvw"));

    std::vector<float> values(12, 0.5F);
    WeightFileWriter writer;
    writer.add("weight", {3, 4}, values.data());
    writer.write(file.path());

    const WeightFile loaded = WeightFile::load(file.path());

    // Same element count, different shape: the check has to compare the shape,
    // not just the size, or a transposed weight would load silently.
    bool caught = false;
    try {
        static_cast<void>(loaded.float_data("weight", {4, 3}));
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()).find("[3, 4]") != std::string::npos;
    }
    require(caught, "a shape mismatch with a matching element count was accepted");

    caught = false;
    try {
        static_cast<void>(loaded.float_data("missing", {1}));
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()).find("missing") != std::string::npos;
    }
    require(caught, "a missing tensor was not reported by name");
}

void test_duplicate_names_are_rejected() {
    std::vector<float> values(4, 1.0F);
    WeightFileWriter writer;
    writer.add("weight", {4}, values.data());

    bool caught = false;
    try {
        writer.add("weight", {4}, values.data());
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "the same tensor name was accepted twice");
}

void test_corrupt_files_are_rejected() {
    bool caught = false;
    try {
        static_cast<void>(WeightFile::load("cuvit_test_does_not_exist.cvw"));
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught, "a missing file was accepted");

    {
        const ScopedFile bad(temporary_path("not_a_weight_file.cvw"));
        std::ofstream stream(bad.path(), std::ios::binary);
        const char junk[] = "this is not a weight file at all";
        stream.write(junk, sizeof(junk));
        stream.close();

        caught = false;
        try {
            static_cast<void>(WeightFile::load(bad.path()));
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "a file with the wrong magic was accepted");
    }

    // A file whose version does not match must be refused rather than
    // reinterpreted, which is the reason the version field exists.
    {
        const ScopedFile good(temporary_path("version.cvw"));
        std::vector<float> values(4, 1.0F);
        WeightFileWriter writer;
        writer.add("weight", {4}, values.data());
        writer.write(good.path());

        std::fstream stream(good.path(), std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(sizeof(std::uint32_t));
        const std::uint32_t wrong_version = WeightFile::kVersion + 1;
        stream.write(reinterpret_cast<const char*>(&wrong_version), sizeof(wrong_version));
        stream.close();

        caught = false;
        try {
            static_cast<void>(WeightFile::load(good.path()));
        } catch (const std::runtime_error& error) {
            caught = std::string(error.what()).find("version") != std::string::npos;
        }
        require(caught, "a file with an unknown version was accepted");
    }

    // Truncation must be caught by the offset check rather than read past the
    // end of the buffer.
    {
        const ScopedFile truncated(temporary_path("truncated.cvw"));
        std::vector<float> values(64, 1.0F);
        WeightFileWriter writer;
        writer.add("weight", {64}, values.data());
        writer.write(truncated.path());

        std::ifstream input(truncated.path(), std::ios::binary);
        std::vector<char> content((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
        input.close();
        content.resize(content.size() - 128);
        std::ofstream output(truncated.path(), std::ios::binary | std::ios::trunc);
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.close();

        caught = false;
        try {
            static_cast<void>(WeightFile::load(truncated.path()));
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "a truncated file was accepted");
    }
}

// The path the engine will actually take: read a file, size an arena from it,
// and upload every tensor.
void test_upload_into_an_arena() {
    const ScopedFile file(temporary_path("arena_upload.cvw"));

    std::vector<float> first(192 * 192);
    std::vector<float> second(192);
    DataGenerator generator(11);
    generator.normal(first.data(), first.size());
    generator.normal(second.data(), second.size());

    WeightFileWriter writer;
    writer.add("proj.weight", {192, 192}, first.data());
    writer.add("proj.bias", {192}, second.data());
    writer.write(file.path());

    const WeightFile loaded = WeightFile::load(file.path());

    std::size_t required = 0;
    for (const cuvit::WeightEntry& entry : loaded.entries()) {
        required += cuvit::DeviceArena::reserved_bytes<float>(
            static_cast<std::size_t>(entry.element_count()));
    }

    cuvit::DeviceArena arena(required);
    std::vector<cuvit::TensorView<float>> uploaded;
    for (const cuvit::WeightEntry& entry : loaded.entries()) {
        cuvit::TensorView<float> view =
            entry.extents.size() == 2 ? arena.allocate<float>({entry.extents[0], entry.extents[1]})
                                      : arena.allocate<float>({entry.extents[0]});
        CUVIT_CUDA_CHECK(cudaMemcpy(view.data(), loaded.float_data(entry.name, entry.extents),
                                    entry.bytes, cudaMemcpyHostToDevice));
        uploaded.push_back(view);
    }

    require(arena.remaining() == 0, "the arena was not sized exactly from the file");

    std::vector<float> readback(first.size());
    CUVIT_CUDA_CHECK(cudaMemcpy(readback.data(), uploaded[0].data(), first.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    cuvit::testing::require_exact(readback.data(), first.data(), first.size(),
                                  "a weight changed on its way to the device");
}

} // namespace

CUVIT_TEST_MAIN("weights_test", {
    test_round_trip_preserves_shape_and_data();
    test_payloads_are_aligned();
    test_shape_mismatch_is_rejected();
    test_duplicate_names_are_rejected();
    test_corrupt_files_are_rejected();
    test_upload_into_an_arena();
})
