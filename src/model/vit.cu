#include "cuvit/vit.hpp"

#include "cuvit/cuda_check.hpp"
#include "cuvit/kernels.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvit {
namespace {

std::string block_prefix(int index) { return "blocks." + std::to_string(index) + '.'; }

/// Uploads one tensor from the file into @p arena, checking its shape first.
const float* upload(DeviceArena& arena, const WeightFile& file, const std::string& name,
                    const std::vector<Extent>& extents) {
    const float* const host = file.float_data(name, extents);
    Extent count = 1;
    for (const Extent extent : extents) {
        count *= extent;
    }
    float* const device = arena.allocate_raw<float>(static_cast<std::size_t>(count));
    CUVIT_CUDA_CHECK(cudaMemcpy(device, host, static_cast<std::size_t>(count) * sizeof(float),
                                cudaMemcpyHostToDevice));
    return device;
}

} // namespace

void VitConfig::validate() const {
    if (image_size <= 0 || patch_size <= 0 || channels <= 0 || embed_dim <= 0 || depth <= 0 ||
        heads <= 0 || mlp_hidden <= 0 || classes <= 0) {
        throw std::invalid_argument("Every ViT dimension must be positive");
    }
    if (image_size % patch_size != 0) {
        throw std::invalid_argument("The image size must divide into whole patches");
    }
    if (embed_dim % heads != 0) {
        throw std::invalid_argument("The embedding dimension must divide across the heads");
    }
}

std::vector<std::string> VisionTransformer::expected_tensor_names(const VitConfig& config) {
    std::vector<std::string> names = {"cls_token", "pos_embed", "patch_embed.proj.weight",
                                      "patch_embed.proj.bias"};
    for (int index = 0; index < config.depth; ++index) {
        const std::string prefix = block_prefix(index);
        for (const char* suffix :
             {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
              "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
              "mlp.fc2.weight", "mlp.fc2.bias"}) {
            names.push_back(prefix + suffix);
        }
    }
    names.insert(names.end(), {"norm.weight", "norm.bias", "head.weight", "head.bias"});
    return names;
}

VisionTransformer::VisionTransformer(const VitConfig& config, const WeightFile& file)
    : config_(config) {
    config_.validate();

    const Extent embed = config_.embed_dim;
    const Extent tokens = config_.token_count();
    const Extent patches = config_.patch_count();
    const Extent hidden = config_.mlp_hidden;
    const Extent patch_elements = config_.patch_elements();
    const Extent classes = config_.classes;

    // Size the weight arena from the shapes rather than from the file, so a file
    // carrying extra tensors does not silently inflate the allocation.
    std::size_t weight_size = 0;
    const auto reserve = [&weight_size](Extent count) {
        weight_size += DeviceArena::reserved_bytes<float>(static_cast<std::size_t>(count));
    };
    reserve(embed);                  // cls_token
    reserve(tokens * embed);         // pos_embed
    reserve(embed * patch_elements); // patch weight
    reserve(embed);                  // patch bias
    for (int index = 0; index < config_.depth; ++index) {
        reserve(embed);             // norm1 gamma
        reserve(embed);             // norm1 beta
        reserve(3 * embed * embed); // qkv weight
        reserve(3 * embed);         // qkv bias
        reserve(embed * embed);     // proj weight
        reserve(embed);             // proj bias
        reserve(embed);             // norm2 gamma
        reserve(embed);             // norm2 beta
        reserve(hidden * embed);    // fc1 weight
        reserve(hidden);            // fc1 bias
        reserve(embed * hidden);    // fc2 weight
        reserve(embed);             // fc2 bias
    }
    reserve(embed);           // final gamma
    reserve(embed);           // final beta
    reserve(classes * embed); // head weight
    reserve(classes);         // head bias

    weights_.reserve(weight_size);

    class_token_ = upload(weights_, file, "cls_token", {1, 1, embed});
    position_embedding_ = upload(weights_, file, "pos_embed", {1, tokens, embed});
    // The convolution weight is used as a flat [embed, patch_elements] matrix.
    // Its memory layout already matches, so no repacking is needed.
    patch_weight_ = upload(weights_, file, "patch_embed.proj.weight",
                           {embed, config_.channels, config_.patch_size, config_.patch_size});
    patch_bias_ = upload(weights_, file, "patch_embed.proj.bias", {embed});

    blocks_.resize(static_cast<std::size_t>(config_.depth));
    for (int index = 0; index < config_.depth; ++index) {
        const std::string prefix = block_prefix(index);
        BlockWeights& block = blocks_[static_cast<std::size_t>(index)];
        block.norm1_gamma = upload(weights_, file, prefix + "norm1.weight", {embed});
        block.norm1_beta = upload(weights_, file, prefix + "norm1.bias", {embed});
        block.qkv_weight = upload(weights_, file, prefix + "attn.qkv.weight", {3 * embed, embed});
        block.qkv_bias = upload(weights_, file, prefix + "attn.qkv.bias", {3 * embed});
        block.proj_weight = upload(weights_, file, prefix + "attn.proj.weight", {embed, embed});
        block.proj_bias = upload(weights_, file, prefix + "attn.proj.bias", {embed});
        block.norm2_gamma = upload(weights_, file, prefix + "norm2.weight", {embed});
        block.norm2_beta = upload(weights_, file, prefix + "norm2.bias", {embed});
        block.fc1_weight = upload(weights_, file, prefix + "mlp.fc1.weight", {hidden, embed});
        block.fc1_bias = upload(weights_, file, prefix + "mlp.fc1.bias", {hidden});
        block.fc2_weight = upload(weights_, file, prefix + "mlp.fc2.weight", {embed, hidden});
        block.fc2_bias = upload(weights_, file, prefix + "mlp.fc2.bias", {embed});
    }

    final_gamma_ = upload(weights_, file, "norm.weight", {embed});
    final_beta_ = upload(weights_, file, "norm.bias", {embed});
    head_weight_ = upload(weights_, file, "head.weight", {classes, embed});
    head_bias_ = upload(weights_, file, "head.bias", {classes});

    // Activation scratch. Every buffer is reused across all blocks, so depth
    // costs no memory.
    std::size_t activation_size = 0;
    const auto reserve_activation = [&activation_size](Extent count) {
        activation_size += DeviceArena::reserved_bytes<float>(static_cast<std::size_t>(count));
    };
    reserve_activation(patches * patch_elements);        // patch matrix
    reserve_activation(tokens * embed);                  // token stream
    reserve_activation(tokens * embed);                  // normalized input
    reserve_activation(tokens * 3 * embed);              // packed qkv
    reserve_activation(config_.heads * tokens * tokens); // attention weights
    reserve_activation(tokens * embed);                  // attention context
    reserve_activation(tokens * embed);                  // projection output
    reserve_activation(tokens * hidden);                 // mlp hidden

    activations_.reserve(activation_size);
    patches_ = activations_.allocate_raw<float>(static_cast<std::size_t>(patches * patch_elements));
    tokens_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * embed));
    normalized_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * embed));
    qkv_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * 3 * embed));
    attention_ =
        activations_.allocate_raw<float>(static_cast<std::size_t>(config_.heads) * tokens * tokens);
    context_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * embed));
    projected_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * embed));
    hidden_ = activations_.allocate_raw<float>(static_cast<std::size_t>(tokens * hidden));
}

void VisionTransformer::run_block(const BlockWeights& block, float* tokens, cudaStream_t stream) {
    const int token_count = config_.token_count();
    const int embed = config_.embed_dim;
    const int heads = config_.heads;
    const int head_dim = config_.head_dim();
    const int hidden = config_.mlp_hidden;
    const std::int64_t token_elements = static_cast<std::int64_t>(token_count) * embed;

    // --- Attention -----------------------------------------------------------
    launch_layer_norm(tokens, block.norm1_gamma, block.norm1_beta, normalized_, token_count, embed,
                      config_.layer_norm_epsilon, stream);

    // One projection produces Q, K and V interleaved as [tokens, 3, heads, head_dim].
    launch_gemm(normalized_, block.qkv_weight, block.qkv_bias, qkv_, token_count, 3 * embed, embed,
                GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);

    const int qkv_row = 3 * embed;
    const float* const queries = qkv_;
    const float* const keys = qkv_ + embed;
    const float* const values = qkv_ + 2 * embed;
    // 1/sqrt(head_dim), the scale that keeps the logits from growing with the
    // head dimension.
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    // Q @ K^T per head. Both operands are column slices of the packed
    // projection, reached by leading dimension and batch stride instead of being
    // gathered into contiguous per-head buffers first.
    launch_gemm(queries, keys, nullptr, attention_, token_count, token_count, head_dim,
                GemmLayout::kTransposed, scale, heads, head_dim, head_dim,
                static_cast<std::int64_t>(token_count) * token_count, qkv_row, qkv_row, token_count,
                stream);

    launch_softmax(attention_, attention_, heads * token_count, token_count, stream);

    // attention @ V, written straight into the [tokens, heads * head_dim] layout
    // the output projection expects, so no transpose is needed afterwards.
    launch_gemm(attention_, values, nullptr, context_, token_count, head_dim, token_count,
                GemmLayout::kNoTranspose, 1.0F, heads,
                static_cast<std::int64_t>(token_count) * token_count, head_dim, head_dim,
                token_count, qkv_row, embed, stream);

    launch_gemm(context_, block.proj_weight, block.proj_bias, projected_, token_count, embed, embed,
                GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);
    launch_add(tokens, projected_, token_elements, stream);

    // --- Feed forward --------------------------------------------------------
    launch_layer_norm(tokens, block.norm2_gamma, block.norm2_beta, normalized_, token_count, embed,
                      config_.layer_norm_epsilon, stream);
    launch_gemm(normalized_, block.fc1_weight, block.fc1_bias, hidden_, token_count, hidden, embed,
                GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);
    launch_gelu(hidden_, hidden_, static_cast<std::int64_t>(token_count) * hidden, stream);
    launch_gemm(hidden_, block.fc2_weight, block.fc2_bias, projected_, token_count, embed, hidden,
                GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);
    launch_add(tokens, projected_, token_elements, stream);
}

void VisionTransformer::forward(const float* image, float* logits, cudaStream_t stream) {
    if (image == nullptr || logits == nullptr) {
        throw std::invalid_argument("Forward pass operands cannot be null");
    }

    const int token_count = config_.token_count();
    const int patch_count = config_.patch_count();
    const int embed = config_.embed_dim;
    const int patch_elements = config_.patch_elements();

    // A patch embedding is a convolution with kernel equal to stride, so it is a
    // gather followed by one GEMM rather than a general convolution.
    launch_extract_patches(image, patches_, config_.channels, config_.image_size,
                           config_.image_size, config_.patch_size, stream);

    // The classification token occupies row zero, so the patch embeddings are
    // written starting at row one and the two never need to be concatenated.
    launch_gemm(patches_, patch_weight_, patch_bias_, tokens_ + embed, patch_count, embed,
                patch_elements, GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);

    CUVIT_CUDA_CHECK(cudaMemcpyAsync(tokens_, class_token_,
                                     static_cast<std::size_t>(embed) * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream));

    launch_add(tokens_, position_embedding_, static_cast<std::int64_t>(token_count) * embed,
               stream);

    for (const BlockWeights& block : blocks_) {
        run_block(block, tokens_, stream);
    }

    launch_layer_norm(tokens_, final_gamma_, final_beta_, normalized_, token_count, embed,
                      config_.layer_norm_epsilon, stream);

    // Only the classification token feeds the head; timm's default pooling for
    // this model is 'token', not a mean over the patches.
    launch_gemm(normalized_, head_weight_, head_bias_, logits, 1, config_.classes, embed,
                GemmLayout::kTransposed, 1.0F, 1, 0, 0, 0, 0, 0, 0, stream);
}

std::vector<float> VisionTransformer::forward(const std::vector<float>& image) {
    const std::size_t expected =
        static_cast<std::size_t>(config_.channels) * config_.image_size * config_.image_size;
    if (image.size() != expected) {
        throw std::invalid_argument("The image has " + std::to_string(image.size()) +
                                    " elements but the model expects " + std::to_string(expected));
    }

    DeviceBuffer<float> device_image(image.size());
    DeviceBuffer<float> device_logits(static_cast<std::size_t>(config_.classes));
    Stream stream;

    device_image.copy_from_host_async(image.data(), image.size(), stream.get());
    forward(device_image.data(), device_logits.data(), stream.get());

    std::vector<float> logits(static_cast<std::size_t>(config_.classes));
    device_logits.copy_to_host_async(logits.data(), logits.size(), stream.get());
    stream.synchronize();
    return logits;
}

} // namespace cuvit
