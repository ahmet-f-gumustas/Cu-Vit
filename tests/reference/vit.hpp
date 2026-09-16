#pragma once

#include "cuvit/vit.hpp"
#include "cuvit/weights.hpp"
#include "reference/linear_algebra.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cuvit::reference {

/// A whole Vision Transformer forward pass on the CPU.
///
/// Written as a transcription of the architecture rather than as an efficient
/// implementation: every step is a separate named buffer, and the loops are the
/// obvious ones. The engine is judged against this, so a shortcut taken here
/// that happens to match a shortcut in a kernel would hide a defect in both.
class VitReference final {
  public:
    VitReference(const VitConfig& config, const WeightFile& file) : config_(config) {
        const Extent embed = config_.embed_dim;
        const Extent tokens = config_.token_count();
        const Extent hidden = config_.mlp_hidden;

        class_token_ = fetch(file, "cls_token", {1, 1, embed});
        position_embedding_ = fetch(file, "pos_embed", {1, tokens, embed});
        patch_weight_ = fetch(file, "patch_embed.proj.weight",
                              {embed, config_.channels, config_.patch_size, config_.patch_size});
        patch_bias_ = fetch(file, "patch_embed.proj.bias", {embed});

        blocks_.resize(static_cast<std::size_t>(config_.depth));
        for (int index = 0; index < config_.depth; ++index) {
            const std::string prefix = "blocks." + std::to_string(index) + '.';
            Block& block = blocks_[static_cast<std::size_t>(index)];
            block.norm1_gamma = fetch(file, prefix + "norm1.weight", {embed});
            block.norm1_beta = fetch(file, prefix + "norm1.bias", {embed});
            block.qkv_weight = fetch(file, prefix + "attn.qkv.weight", {3 * embed, embed});
            block.qkv_bias = fetch(file, prefix + "attn.qkv.bias", {3 * embed});
            block.proj_weight = fetch(file, prefix + "attn.proj.weight", {embed, embed});
            block.proj_bias = fetch(file, prefix + "attn.proj.bias", {embed});
            block.norm2_gamma = fetch(file, prefix + "norm2.weight", {embed});
            block.norm2_beta = fetch(file, prefix + "norm2.bias", {embed});
            block.fc1_weight = fetch(file, prefix + "mlp.fc1.weight", {hidden, embed});
            block.fc1_bias = fetch(file, prefix + "mlp.fc1.bias", {hidden});
            block.fc2_weight = fetch(file, prefix + "mlp.fc2.weight", {embed, hidden});
            block.fc2_bias = fetch(file, prefix + "mlp.fc2.bias", {embed});
        }

        final_gamma_ = fetch(file, "norm.weight", {embed});
        final_beta_ = fetch(file, "norm.bias", {embed});
        head_weight_ = fetch(file, "head.weight", {config_.classes, embed});
        head_bias_ = fetch(file, "head.bias", {config_.classes});
    }

    [[nodiscard]] std::vector<float> forward(const std::vector<float>& image) const {
        const int embed = config_.embed_dim;
        const int token_count = config_.token_count();
        const int patch_count = config_.patch_count();
        const int patch_elements = config_.patch_elements();
        const int patch = config_.patch_size;
        const int across = config_.patches_per_side();

        // Patch extraction: a convolution whose kernel equals its stride visits
        // each input element once, so it is a gather into rows.
        std::vector<float> patches(static_cast<std::size_t>(patch_count) * patch_elements);
        for (int index = 0; index < patch_count; ++index) {
            const int patch_row = index / across;
            const int patch_column = index % across;
            for (int channel = 0; channel < config_.channels; ++channel) {
                for (int row = 0; row < patch; ++row) {
                    for (int column = 0; column < patch; ++column) {
                        const int source_row = patch_row * patch + row;
                        const int source_column = patch_column * patch + column;
                        const std::size_t destination =
                            static_cast<std::size_t>(index) * patch_elements +
                            (static_cast<std::size_t>(channel) * patch + row) * patch + column;
                        patches[destination] =
                            image[(static_cast<std::size_t>(channel) * config_.image_size +
                                   source_row) *
                                      config_.image_size +
                                  source_column];
                    }
                }
            }
        }

        std::vector<float> tokens(static_cast<std::size_t>(token_count) * embed);
        for (int index = 0; index < embed; ++index) {
            tokens[static_cast<std::size_t>(index)] = class_token_[index];
        }
        gemm_transposed(patches.data(), patch_weight_.data(), patch_bias_.data(),
                        tokens.data() + embed, patch_count, embed, patch_elements);

        for (std::size_t index = 0; index < tokens.size(); ++index) {
            tokens[index] += position_embedding_[index];
        }

        for (const Block& block : blocks_) {
            run_block(block, tokens);
        }

        std::vector<float> normalized(tokens.size());
        layer_norm(tokens.data(), final_gamma_.data(), final_beta_.data(), normalized.data(),
                   token_count, embed, config_.layer_norm_epsilon);

        std::vector<float> logits(static_cast<std::size_t>(config_.classes));
        gemm_transposed(normalized.data(), head_weight_.data(), head_bias_.data(), logits.data(), 1,
                        config_.classes, embed);
        return logits;
    }

  private:
    struct Block {
        std::vector<float> norm1_gamma, norm1_beta;
        std::vector<float> qkv_weight, qkv_bias;
        std::vector<float> proj_weight, proj_bias;
        std::vector<float> norm2_gamma, norm2_beta;
        std::vector<float> fc1_weight, fc1_bias;
        std::vector<float> fc2_weight, fc2_bias;
    };

    static std::vector<float> fetch(const WeightFile& file, const std::string& name,
                                    const std::vector<Extent>& extents) {
        const float* const data = file.float_data(name, extents);
        Extent count = 1;
        for (const Extent extent : extents) {
            count *= extent;
        }
        return std::vector<float>(data, data + count);
    }

    void run_block(const Block& block, std::vector<float>& tokens) const {
        const int token_count = config_.token_count();
        const int embed = config_.embed_dim;
        const int heads = config_.heads;
        const int head_dim = config_.head_dim();
        const int hidden = config_.mlp_hidden;

        std::vector<float> normalized(tokens.size());
        layer_norm(tokens.data(), block.norm1_gamma.data(), block.norm1_beta.data(),
                   normalized.data(), token_count, embed, config_.layer_norm_epsilon);

        std::vector<float> qkv(static_cast<std::size_t>(token_count) * 3 * embed);
        gemm_transposed(normalized.data(), block.qkv_weight.data(), block.qkv_bias.data(),
                        qkv.data(), token_count, 3 * embed, embed);

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        const int row = 3 * embed;

        std::vector<float> context(static_cast<std::size_t>(token_count) * embed);
        std::vector<float> logits(static_cast<std::size_t>(token_count) * token_count);
        std::vector<float> weights(logits.size());

        for (int head = 0; head < heads; ++head) {
            for (int query = 0; query < token_count; ++query) {
                for (int key = 0; key < token_count; ++key) {
                    float sum = 0.0F;
                    for (int index = 0; index < head_dim; ++index) {
                        const float q =
                            qkv[static_cast<std::size_t>(query) * row + head * head_dim + index];
                        const float k = qkv[static_cast<std::size_t>(key) * row + embed +
                                            head * head_dim + index];
                        sum += q * k;
                    }
                    logits[static_cast<std::size_t>(query) * token_count + key] = sum * scale;
                }
            }

            softmax(logits.data(), weights.data(), token_count, token_count);

            for (int query = 0; query < token_count; ++query) {
                for (int index = 0; index < head_dim; ++index) {
                    float sum = 0.0F;
                    for (int key = 0; key < token_count; ++key) {
                        const float v = qkv[static_cast<std::size_t>(key) * row + 2 * embed +
                                            head * head_dim + index];
                        sum += weights[static_cast<std::size_t>(query) * token_count + key] * v;
                    }
                    context[static_cast<std::size_t>(query) * embed + head * head_dim + index] =
                        sum;
                }
            }
        }

        std::vector<float> projected(tokens.size());
        gemm_transposed(context.data(), block.proj_weight.data(), block.proj_bias.data(),
                        projected.data(), token_count, embed, embed);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            tokens[index] += projected[index];
        }

        layer_norm(tokens.data(), block.norm2_gamma.data(), block.norm2_beta.data(),
                   normalized.data(), token_count, embed, config_.layer_norm_epsilon);

        std::vector<float> hidden_values(static_cast<std::size_t>(token_count) * hidden);
        gemm_transposed(normalized.data(), block.fc1_weight.data(), block.fc1_bias.data(),
                        hidden_values.data(), token_count, hidden, embed);
        gelu(hidden_values.data(), hidden_values.data(), hidden_values.size());
        gemm_transposed(hidden_values.data(), block.fc2_weight.data(), block.fc2_bias.data(),
                        projected.data(), token_count, embed, hidden);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            tokens[index] += projected[index];
        }
    }

    VitConfig config_;
    std::vector<float> class_token_, position_embedding_;
    std::vector<float> patch_weight_, patch_bias_;
    std::vector<Block> blocks_;
    std::vector<float> final_gamma_, final_beta_;
    std::vector<float> head_weight_, head_bias_;
};

} // namespace cuvit::reference
