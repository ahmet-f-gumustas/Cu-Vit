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

std::size_t VisionTransformer::image_elements() const noexcept {
    return static_cast<std::size_t>(config_.channels) * config_.image_size * config_.image_size;
}

VisionTransformer::VisionTransformer(const VitConfig& config, const WeightFile& file, int max_batch)
    : config_(config), max_batch_(max_batch) {
    config_.validate();
    if (max_batch_ < 1) {
        throw std::invalid_argument("The maximum batch size must be at least one");
    }

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

    // Activation scratch, sized for the largest batch this model will run.
    // Every buffer is reused across all blocks, so depth costs no memory.
    const Extent images = max_batch_;
    std::size_t activation_size = 0;
    const auto reserve_activation = [&activation_size](Extent count) {
        activation_size += DeviceArena::reserved_bytes<float>(static_cast<std::size_t>(count));
    };
    reserve_activation(images * patches * patch_elements);        // patch matrix
    reserve_activation(images * tokens * embed);                  // token stream
    reserve_activation(images * tokens * embed);                  // normalized input
    reserve_activation(images * tokens * 3 * embed);              // packed qkv
    reserve_activation(images * config_.heads * tokens * tokens); // attention weights
    reserve_activation(images * tokens * embed);                  // attention context
    reserve_activation(images * tokens * hidden);                 // mlp hidden

    activations_.reserve(activation_size);
    const auto carve = [this](Extent count) {
        return activations_.allocate_raw<float>(static_cast<std::size_t>(count));
    };
    patches_ = carve(images * patches * patch_elements);
    tokens_ = carve(images * tokens * embed);
    normalized_ = carve(images * tokens * embed);
    qkv_ = carve(images * tokens * 3 * embed);
    attention_ = carve(images * config_.heads * tokens * tokens);
    context_ = carve(images * tokens * embed);
    hidden_ = carve(images * tokens * hidden);
}

void VisionTransformer::run_block(const BlockWeights& block, float* tokens, int batch,
                                  cudaStream_t stream) {
    const int token_count = config_.token_count();
    const int embed = config_.embed_dim;
    const int heads = config_.heads;
    const int head_dim = config_.head_dim();
    const int hidden = config_.mlp_hidden;

    // The token rows of a batch are contiguous, so every GEMM that works per
    // token simply sees more rows. Only attention, which is per image and per
    // head, needs the batch spelled out.
    const int rows = batch * token_count;
    const std::int64_t tokens_per_image = static_cast<std::int64_t>(token_count) * embed;
    const std::int64_t qkv_per_image = static_cast<std::int64_t>(token_count) * 3 * embed;
    const std::int64_t attention_per_head = static_cast<std::int64_t>(token_count) * token_count;
    const int qkv_row = 3 * embed;

    // --- Attention -----------------------------------------------------------
    launch_layer_norm(tokens, block.norm1_gamma, block.norm1_beta, normalized_, rows, embed,
                      config_.layer_norm_epsilon, stream);

    // One projection produces Q, K and V interleaved as [tokens, 3, heads, head_dim].
    launch_gemm(normalized_, block.qkv_weight, block.qkv_bias, qkv_,
                gemm_problem(rows, 3 * embed, embed, GemmLayout::kTransposed), stream);

    const float* const queries = qkv_;
    const float* const keys = qkv_ + embed;
    const float* const values = qkv_ + 2 * embed;
    // 1/sqrt(head_dim), the scale that keeps the logits from growing with the
    // head dimension.
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    // Q @ K^T per image and per head. Both operands are column slices of the
    // packed projection, reached by leading dimension and batch stride rather
    // than gathered into per-head buffers first. The batch index splits as
    // (image, head) because one head of one image sits at
    // image * qkv_per_image + head * head_dim, which no single stride reaches.
    GemmProblem logits = gemm_problem(token_count, token_count, head_dim, GemmLayout::kTransposed);
    logits.alpha = scale;
    logits.lda = qkv_row;
    logits.ldb = qkv_row;
    logits.ldc = token_count;
    logits.batch.count = batch * heads;
    logits.batch.inner = heads;
    logits.batch.stride_a = head_dim;
    logits.batch.stride_b = head_dim;
    logits.batch.stride_c = attention_per_head;
    logits.batch.outer_stride_a = qkv_per_image;
    logits.batch.outer_stride_b = qkv_per_image;
    logits.batch.outer_stride_c = heads * attention_per_head;
    launch_gemm(queries, keys, nullptr, attention_, logits, stream);

    launch_softmax(attention_, attention_, batch * heads * token_count, token_count, stream);

    // attention @ V, written straight into the [tokens, heads * head_dim] layout
    // the output projection expects, so no transpose is needed afterwards.
    GemmProblem context =
        gemm_problem(token_count, head_dim, token_count, GemmLayout::kNoTranspose);
    context.lda = token_count;
    context.ldb = qkv_row;
    context.ldc = embed;
    context.batch.count = batch * heads;
    context.batch.inner = heads;
    context.batch.stride_a = attention_per_head;
    context.batch.stride_b = head_dim;
    context.batch.stride_c = head_dim;
    context.batch.outer_stride_a = heads * attention_per_head;
    context.batch.outer_stride_b = qkv_per_image;
    context.batch.outer_stride_c = tokens_per_image;
    launch_gemm(attention_, values, nullptr, context_, context, stream);

    // The projection accumulates straight into the residual stream, so the
    // addition costs neither a launch nor a pass over the tokens.
    GemmProblem projection = gemm_problem(rows, embed, embed, GemmLayout::kTransposed);
    projection.epilogue.accumulate = true;
    launch_gemm(context_, block.proj_weight, block.proj_bias, tokens, projection, stream);

    // --- Feed forward --------------------------------------------------------
    launch_layer_norm(tokens, block.norm2_gamma, block.norm2_beta, normalized_, rows, embed,
                      config_.layer_norm_epsilon, stream);

    GemmProblem first = gemm_problem(rows, hidden, embed, GemmLayout::kTransposed);
    first.epilogue.gelu = true;
    launch_gemm(normalized_, block.fc1_weight, block.fc1_bias, hidden_, first, stream);

    GemmProblem second = gemm_problem(rows, embed, hidden, GemmLayout::kTransposed);
    second.epilogue.accumulate = true;
    launch_gemm(hidden_, block.fc2_weight, block.fc2_bias, tokens, second, stream);
}

void VisionTransformer::forward(const float* images, float* logits, int batch,
                                cudaStream_t stream) {
    if (batch == 0) {
        return;
    }
    if (batch < 0 || batch > max_batch_) {
        throw std::invalid_argument("This model was built for at most " +
                                    std::to_string(max_batch_) + " images, but " +
                                    std::to_string(batch) + " were given");
    }
    if (images == nullptr || logits == nullptr) {
        throw std::invalid_argument("Forward pass operands cannot be null");
    }

    const int token_count = config_.token_count();
    const int patch_count = config_.patch_count();
    const int embed = config_.embed_dim;
    const int patch_elements = config_.patch_elements();
    const std::int64_t tokens_per_image = static_cast<std::int64_t>(token_count) * embed;

    // A patch embedding is a convolution with kernel equal to stride, so it is a
    // gather followed by one GEMM rather than a general convolution.
    launch_extract_patches(images, patches_, batch, config_.channels, config_.image_size,
                           config_.image_size, config_.patch_size, stream);

    // The class token occupies row zero of each image, so the patch embeddings
    // are written starting at row one and the two never need concatenating.
    GemmProblem embedding =
        gemm_problem(patch_count, embed, patch_elements, GemmLayout::kTransposed);
    embedding.batch.count = batch;
    embedding.batch.stride_a = static_cast<std::int64_t>(patch_count) * patch_elements;
    embedding.batch.stride_c = tokens_per_image;
    launch_gemm(patches_, patch_weight_, patch_bias_, tokens_ + embed, embedding, stream);

    // One stored class token and one stored positional embedding, broadcast
    // across the batch rather than replicated in memory.
    launch_broadcast(tokens_, class_token_, embed, tokens_per_image, batch, stream);
    launch_add_broadcast(tokens_, position_embedding_, tokens_per_image, batch, stream);

    for (const BlockWeights& block : blocks_) {
        run_block(block, tokens_, batch, stream);
    }

    launch_layer_norm(tokens_, final_gamma_, final_beta_, normalized_, batch * token_count, embed,
                      config_.layer_norm_epsilon, stream);

    // Only the class token feeds the head; timm's default pooling for this model
    // is 'token', not a mean over the patches. One row per image, so the batch is
    // expressed as a stride rather than as extra rows.
    GemmProblem head = gemm_problem(1, config_.classes, embed, GemmLayout::kTransposed);
    head.batch.count = batch;
    head.batch.stride_a = tokens_per_image;
    head.batch.stride_c = config_.classes;
    launch_gemm(normalized_, head_weight_, head_bias_, logits, head, stream);
}

std::vector<float> VisionTransformer::forward(const std::vector<float>& images) {
    const std::size_t per_image = image_elements();
    if (per_image == 0 || images.size() % per_image != 0) {
        throw std::invalid_argument("The input holds " + std::to_string(images.size()) +
                                    " elements, which is not a whole number of " +
                                    std::to_string(per_image) + "-element images");
    }
    const std::size_t batch = images.size() / per_image;
    if (batch > static_cast<std::size_t>(max_batch_)) {
        throw std::invalid_argument("This model was built for at most " +
                                    std::to_string(max_batch_) + " images, but " +
                                    std::to_string(batch) + " were given");
    }

    DeviceBuffer<float> device_images(images.size());
    DeviceBuffer<float> device_logits(batch * static_cast<std::size_t>(config_.classes));
    Stream stream;

    device_images.copy_from_host_async(images.data(), images.size(), stream.get());
    forward(device_images.data(), device_logits.data(), static_cast<int>(batch), stream.get());

    std::vector<float> logits(batch * static_cast<std::size_t>(config_.classes));
    device_logits.copy_to_host_async(logits.data(), logits.size(), stream.get());
    stream.synchronize();
    return logits;
}

} // namespace cuvit
