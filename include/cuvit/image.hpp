#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuvit {

/// An 8-bit RGB image in interleaved order.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels; // width * height * 3

    [[nodiscard]] bool empty() const noexcept { return pixels.empty(); }
};

/// Reads a binary PPM (P6).
///
/// PPM is the one uncompressed format that needs no third-party decoder, so the
/// engine can be exercised on real images without taking a dependency. Convert
/// anything else first:
///
///     ffmpeg -i photo.jpg -pix_fmt rgb24 photo.ppm
[[nodiscard]] RgbImage load_ppm(const std::string& path);

/// How a model's preprocessing maps pixels to tensor values.
struct Preprocessing {
    int size = 224;
    /// Fraction of the shorter side kept before the centre crop. timm reports
    /// this as crop_pct; 0.9 means resize so that 224 / 0.9 = 248 pixels of the
    /// shorter side survive, then crop the middle 224.
    float crop_fraction = 0.9F;
    std::array<float, 3> mean = {0.5F, 0.5F, 0.5F};
    std::array<float, 3> stddev = {0.5F, 0.5F, 0.5F};
};

/// Turns an image into the [channels, size, size] tensor the model expects.
///
/// Resizes the shorter side to size / crop_fraction preserving aspect ratio,
/// takes the centre crop, converts to the [0, 1] range, then normalizes by mean
/// and standard deviation. The result is planar, matching the layout the patch
/// extraction kernel reads.
///
/// Resampling is bilinear, while timm's default for this model is bicubic. The
/// two disagree enough to move a borderline classification, so a score produced
/// here will not match torchvision's to the last digit; the difference is in the
/// resampling, not in the model.
[[nodiscard]] std::vector<float> preprocess(const RgbImage& image,
                                            const Preprocessing& settings = {});

} // namespace cuvit
