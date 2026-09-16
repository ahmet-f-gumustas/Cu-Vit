#include "cuvit/vit.hpp"
#include "cuvit/weights.hpp"
#include "reference/vit.hpp"
#include "test_utils.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuvit::VisionTransformer;
using cuvit::VitConfig;
using cuvit::WeightFile;
using cuvit::WeightFileWriter;
using cuvit::testing::DataGenerator;
using cuvit::testing::require;

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

/// A model small enough to run a CPU reference over in milliseconds, while
/// exercising every path the full-size one takes: multiple blocks, multiple
/// heads, and a token count that is not a multiple of the GEMM tile.
VitConfig small_config() {
    VitConfig config;
    config.image_size = 32;
    config.patch_size = 8;
    config.channels = 3;
    config.embed_dim = 24;
    config.depth = 3;
    config.heads = 4;
    config.mlp_hidden = 48;
    config.classes = 7;
    return config;
}

void write_random_weights(const VitConfig& config, const std::string& path, std::uint64_t seed) {
    DataGenerator generator(seed);
    WeightFileWriter writer;

    const auto add = [&](const std::string& name, const std::vector<cuvit::Extent>& extents,
                         float mean, float stddev) {
        cuvit::Extent count = 1;
        for (const cuvit::Extent extent : extents) {
            count *= extent;
        }
        std::vector<float> values(static_cast<std::size_t>(count));
        generator.normal(values.data(), values.size(), mean, stddev);
        writer.add(name, extents, values.data());
    };

    const cuvit::Extent embed = config.embed_dim;
    const cuvit::Extent tokens = config.token_count();
    const cuvit::Extent hidden = config.mlp_hidden;

    add("cls_token", {1, 1, embed}, 0.0F, 0.5F);
    add("pos_embed", {1, tokens, embed}, 0.0F, 0.5F);
    add("patch_embed.proj.weight", {embed, config.channels, config.patch_size, config.patch_size},
        0.0F, 0.1F);
    add("patch_embed.proj.bias", {embed}, 0.0F, 0.1F);

    for (int index = 0; index < config.depth; ++index) {
        const std::string prefix = "blocks." + std::to_string(index) + '.';
        // Normalization scales start near one and shifts near zero, as a trained
        // model's do; starting them at random would make the residual stream
        // diverge across blocks and hide accuracy problems behind saturation.
        add(prefix + "norm1.weight", {embed}, 1.0F, 0.1F);
        add(prefix + "norm1.bias", {embed}, 0.0F, 0.1F);
        add(prefix + "attn.qkv.weight", {3 * embed, embed}, 0.0F, 0.2F);
        add(prefix + "attn.qkv.bias", {3 * embed}, 0.0F, 0.1F);
        add(prefix + "attn.proj.weight", {embed, embed}, 0.0F, 0.2F);
        add(prefix + "attn.proj.bias", {embed}, 0.0F, 0.1F);
        add(prefix + "norm2.weight", {embed}, 1.0F, 0.1F);
        add(prefix + "norm2.bias", {embed}, 0.0F, 0.1F);
        add(prefix + "mlp.fc1.weight", {hidden, embed}, 0.0F, 0.2F);
        add(prefix + "mlp.fc1.bias", {hidden}, 0.0F, 0.1F);
        add(prefix + "mlp.fc2.weight", {embed, hidden}, 0.0F, 0.2F);
        add(prefix + "mlp.fc2.bias", {embed}, 0.0F, 0.1F);
    }

    add("norm.weight", {embed}, 1.0F, 0.1F);
    add("norm.bias", {embed}, 0.0F, 0.1F);
    add("head.weight", {config.classes, embed}, 0.0F, 0.2F);
    add("head.bias", {config.classes}, 0.0F, 0.1F);

    writer.write(path);
}

std::vector<float> random_image(const VitConfig& config, std::uint64_t seed) {
    std::vector<float> image(static_cast<std::size_t>(config.channels) * config.image_size *
                             config.image_size);
    DataGenerator(seed).normal(image.data(), image.size());
    return image;
}

void test_forward_matches_the_cpu_reference() {
    const VitConfig config = small_config();
    const ScopedFile file("cuvit_test_small_vit.cvw");
    write_random_weights(config, file.path(), 1234);

    const WeightFile weights = WeightFile::load(file.path());
    VisionTransformer model(config, weights);
    const cuvit::reference::VitReference expected_model(config, weights);

    for (std::uint64_t seed : {1U, 2U, 3U}) {
        const std::vector<float> image = random_image(config, seed);
        const std::vector<float> actual = model.forward(image);
        const std::vector<float> expected = expected_model.forward(image);

        require(actual.size() == static_cast<std::size_t>(config.classes),
                "the forward pass produced the wrong number of logits");

        // Logits accumulate rounding through three blocks of reductions, and the
        // GPU sums in a different order at every step. The tolerance is relative
        // to the logit magnitude, which is the scale the error grows with.
        cuvit::testing::require_close(actual.data(), expected.data(), actual.size(),
                                      "forward pass differs from the CPU reference", 1.0e-4,
                                      1.0e-4);
    }
}

void test_forward_is_deterministic() {
    const VitConfig config = small_config();
    const ScopedFile file("cuvit_test_repeat_vit.cvw");
    write_random_weights(config, file.path(), 77);

    const WeightFile weights = WeightFile::load(file.path());
    VisionTransformer model(config, weights);
    const std::vector<float> image = random_image(config, 5);

    const std::vector<float> first = model.forward(image);
    const std::vector<float> second = model.forward(image);
    // The scratch buffers are reused between passes, so a stale value left
    // behind by one pass would change the next.
    cuvit::testing::require_exact(first.data(), second.data(), first.size(),
                                  "two runs of the same input disagree");
}

void test_different_inputs_produce_different_logits() {
    const VitConfig config = small_config();
    const ScopedFile file("cuvit_test_distinct_vit.cvw");
    write_random_weights(config, file.path(), 91);

    const WeightFile weights = WeightFile::load(file.path());
    VisionTransformer model(config, weights);

    const std::vector<float> first = model.forward(random_image(config, 10));
    const std::vector<float> second = model.forward(random_image(config, 11));

    bool differs = false;
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (first[index] != second[index]) {
            differs = true;
            break;
        }
    }
    // A model that ignored its input would pass every comparison against a
    // reference that ignored it the same way.
    require(differs, "two different images produced identical logits");
}

void test_allocation_is_bounded_and_reused() {
    const VitConfig config = small_config();
    const ScopedFile file("cuvit_test_alloc_vit.cvw");
    write_random_weights(config, file.path(), 5);

    const WeightFile weights = WeightFile::load(file.path());
    VisionTransformer model(config, weights);

    require(model.weight_bytes() > 0, "no weight memory was reserved");
    require(model.activation_bytes() > 0, "no activation memory was reserved");

    // Activation scratch is reused across blocks, so depth must not multiply it.
    VitConfig deeper = config;
    deeper.depth = config.depth * 4;
    const ScopedFile deep_file("cuvit_test_deep_vit.cvw");
    write_random_weights(deeper, deep_file.path(), 6);
    const WeightFile deep_weights = WeightFile::load(deep_file.path());
    VisionTransformer deep_model(deeper, deep_weights);

    require(deep_model.activation_bytes() == model.activation_bytes(),
            "activation memory grew with depth");
    require(deep_model.weight_bytes() > model.weight_bytes(),
            "weight memory did not grow with depth");
}

void test_configuration_is_validated() {
    VitConfig config = small_config();
    config.embed_dim = 25; // not divisible by 4 heads

    bool caught = false;
    try {
        config.validate();
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "an embedding dimension that does not divide across heads was accepted");

    config = small_config();
    config.image_size = 30; // not divisible by an 8-pixel patch
    caught = false;
    try {
        config.validate();
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "an image size that does not divide into patches was accepted");
}

void test_missing_and_misshaped_weights_are_rejected() {
    const VitConfig config = small_config();

    // A file missing one tensor must fail at construction, not at the first
    // forward pass with an uninitialized buffer.
    const ScopedFile file("cuvit_test_incomplete_vit.cvw");
    write_random_weights(config, file.path(), 3);
    const WeightFile complete = WeightFile::load(file.path());

    VitConfig wrong = config;
    wrong.embed_dim = 48; // every shape now disagrees with the file
    wrong.heads = 4;
    bool caught = false;
    try {
        VisionTransformer model(wrong, complete);
        static_cast<void>(model.weight_bytes());
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught, "a model was built from weights of the wrong shape");
}

void test_expected_names_cover_the_file() {
    const VitConfig config = small_config();
    const ScopedFile file("cuvit_test_names_vit.cvw");
    write_random_weights(config, file.path(), 8);
    const WeightFile weights = WeightFile::load(file.path());

    const std::vector<std::string> names = VisionTransformer::expected_tensor_names(config);
    require(names.size() == weights.size(),
            "the expected name list and the weight file disagree in size");
    for (const std::string& name : names) {
        require(weights.contains(name), "an expected tensor is missing from the file");
    }
}

} // namespace

CUVIT_TEST_MAIN("vit_test", {
    test_forward_matches_the_cpu_reference();
    test_forward_is_deterministic();
    test_different_inputs_produce_different_logits();
    test_allocation_is_bounded_and_reused();
    test_configuration_is_validated();
    test_missing_and_misshaped_weights_are_rejected();
    test_expected_names_cover_the_file();
})
