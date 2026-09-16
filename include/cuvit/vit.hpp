#pragma once

#include "cuvit/device_arena.hpp"
#include "cuvit/device_buffer.hpp"
#include "cuvit/stream.hpp"
#include "cuvit/tensor.hpp"
#include "cuvit/weights.hpp"

#include <string>
#include <vector>

namespace cuvit {

/// Shape of a Vision Transformer. The defaults describe ViT-Tiny/16 at 224x224.
struct VitConfig {
    int image_size = 224;
    int patch_size = 16;
    int channels = 3;
    int embed_dim = 192;
    int depth = 12;
    int heads = 3;
    int mlp_hidden = 768;
    int classes = 1000;
    float layer_norm_epsilon = 1.0e-6F;

    [[nodiscard]] int patches_per_side() const { return image_size / patch_size; }
    [[nodiscard]] int patch_count() const { return patches_per_side() * patches_per_side(); }
    /// Patches plus the classification token.
    [[nodiscard]] int token_count() const { return patch_count() + 1; }
    [[nodiscard]] int head_dim() const { return embed_dim / heads; }
    [[nodiscard]] int patch_elements() const { return channels * patch_size * patch_size; }

    /// Throws when the configuration cannot describe a model, naming the field.
    void validate() const;
};

/// Device-resident weights for one transformer block.
struct BlockWeights {
    const float* norm1_gamma = nullptr;
    const float* norm1_beta = nullptr;
    const float* qkv_weight = nullptr;
    const float* qkv_bias = nullptr;
    const float* proj_weight = nullptr;
    const float* proj_bias = nullptr;
    const float* norm2_gamma = nullptr;
    const float* norm2_beta = nullptr;
    const float* fc1_weight = nullptr;
    const float* fc1_bias = nullptr;
    const float* fc2_weight = nullptr;
    const float* fc2_bias = nullptr;
};

/// A ViT ready to run: weights uploaded, scratch space reserved.
///
/// Everything is allocated once at construction, sized for @p max_batch images.
/// A forward pass then performs no allocation at all, so its cost is the
/// arithmetic and nothing else.
///
/// Batching is what turns this model from launch-bound into compute-bound. At
/// one image the largest GEMM is [197, 768], too small to occupy the GPU; the
/// token rows of a batch concatenate, so eight images give the same kernels
/// eight times the rows at the same launch count.
class VisionTransformer final {
  public:
    /// Uploads @p file into device memory, checking every tensor's shape against
    /// @p config, and reserves activation space for @p max_batch images. Names
    /// follow timm's state dict.
    VisionTransformer(const VitConfig& config, const WeightFile& file, int max_batch = 1);

    /// Runs @p batch images, given as @p batch planar [channels, height, width]
    /// blocks back to back in device memory, and writes
    /// @p batch x @p config.classes logits to @p logits, also in device memory.
    ///
    /// Work is queued on @p stream; the caller synchronizes before reading.
    void forward(const float* images, float* logits, int batch, cudaStream_t stream = nullptr);

    /// Convenience wrapper taking and returning host memory, synchronizing
    /// internally. The batch size is taken from the size of @p images.
    [[nodiscard]] std::vector<float> forward(const std::vector<float>& images);

    [[nodiscard]] const VitConfig& config() const noexcept { return config_; }
    [[nodiscard]] int max_batch() const noexcept { return max_batch_; }
    /// Elements one image occupies on the input side.
    [[nodiscard]] std::size_t image_elements() const noexcept;

    /// Bytes held for weights and for activations.
    [[nodiscard]] std::size_t weight_bytes() const noexcept { return weights_.used(); }
    [[nodiscard]] std::size_t activation_bytes() const noexcept { return activations_.used(); }

    /// The timm tensor names this configuration expects, in load order.
    [[nodiscard]] static std::vector<std::string> expected_tensor_names(const VitConfig& config);

  private:
    void run_block(const BlockWeights& block, float* tokens, int batch, cudaStream_t stream);

    VitConfig config_;
    int max_batch_ = 1;

    DeviceArena weights_;
    DeviceArena activations_;

    const float* class_token_ = nullptr;
    const float* position_embedding_ = nullptr;
    const float* patch_weight_ = nullptr;
    const float* patch_bias_ = nullptr;
    const float* final_gamma_ = nullptr;
    const float* final_beta_ = nullptr;
    const float* head_weight_ = nullptr;
    const float* head_bias_ = nullptr;
    std::vector<BlockWeights> blocks_;

    // Activation scratch, all carved from activations_.
    float* patches_ = nullptr;
    float* tokens_ = nullptr;
    float* normalized_ = nullptr;
    float* qkv_ = nullptr;
    float* attention_ = nullptr;
    float* context_ = nullptr;
    float* hidden_ = nullptr;
};

} // namespace cuvit
