#include "cuvit/image.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuvit::Preprocessing;
using cuvit::RgbImage;
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

void write_ppm(const std::string& path, int width, int height,
               const std::vector<std::uint8_t>& pixels, const char* comment = nullptr) {
    std::ofstream stream(path, std::ios::binary);
    stream << "P6\n";
    if (comment != nullptr) {
        stream << '#' << comment << '\n';
    }
    stream << width << ' ' << height << "\n255\n";
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));
}

std::vector<std::uint8_t> gradient(int width, int height) {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const std::size_t base = (static_cast<std::size_t>(row) * width + column) * 3;
            pixels[base + 0] = static_cast<std::uint8_t>(column * 255 / (width - 1));
            pixels[base + 1] = static_cast<std::uint8_t>(row * 255 / (height - 1));
            pixels[base + 2] = 128;
        }
    }
    return pixels;
}

void test_ppm_round_trip() {
    const ScopedFile file("cuvit_test_image.ppm");
    const std::vector<std::uint8_t> pixels = gradient(7, 5);
    // A comment between the header fields is legal and a naive parser drops it
    // into the dimensions.
    write_ppm(file.path(), 7, 5, pixels, " written by the test suite");

    const RgbImage image = cuvit::load_ppm(file.path());
    require(image.width == 7 && image.height == 5, "the PPM header was misread");
    require(image.pixels.size() == pixels.size(), "the payload has the wrong size");
    cuvit::testing::require_exact(image.pixels.data(), pixels.data(), pixels.size(),
                                  "the PPM payload changed on the way through");
}

void test_malformed_images_are_rejected() {
    bool caught = false;
    try {
        static_cast<void>(cuvit::load_ppm("cuvit_test_absent.ppm"));
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught, "a missing image was accepted");

    {
        // P3 is the ASCII variant; accepting it would read text as pixels.
        const ScopedFile ascii("cuvit_test_ascii.ppm");
        std::ofstream stream(ascii.path());
        stream << "P3\n2 2\n255\n0 0 0 0 0 0 0 0 0 0 0 0\n";
        stream.close();

        caught = false;
        try {
            static_cast<void>(cuvit::load_ppm(ascii.path()));
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "an ASCII PPM was accepted as binary");
    }

    {
        const ScopedFile truncated("cuvit_test_truncated.ppm");
        write_ppm(truncated.path(), 8, 8, std::vector<std::uint8_t>(8 * 8 * 3, 0));
        std::ifstream input(truncated.path(), std::ios::binary);
        std::vector<char> content((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
        input.close();
        content.resize(content.size() - 32);
        std::ofstream output(truncated.path(), std::ios::binary | std::ios::trunc);
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.close();

        caught = false;
        try {
            static_cast<void>(cuvit::load_ppm(truncated.path()));
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "a truncated image was accepted");
    }
}

void test_preprocess_shape_and_normalization() {
    RgbImage image;
    image.width = 300;
    image.height = 300;
    // A uniform mid-grey maps exactly onto the mean, so every output must be
    // zero regardless of how the resampling lands.
    image.pixels.assign(static_cast<std::size_t>(300) * 300 * 3, 128);

    Preprocessing settings;
    const std::vector<float> tensor = cuvit::preprocess(image, settings);
    require(tensor.size() == static_cast<std::size_t>(3) * settings.size * settings.size,
            "the preprocessed tensor has the wrong size");

    const float expected = (128.0F / 255.0F - 0.5F) / 0.5F;
    for (const float value : tensor) {
        require(std::abs(value - expected) < 1.0e-5F,
                "a uniform image did not normalize to a uniform value");
    }
}

void test_preprocess_is_planar_and_keeps_channels_apart() {
    RgbImage image;
    image.width = 256;
    image.height = 256;
    image.pixels.resize(static_cast<std::size_t>(256) * 256 * 3);
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(256) * 256; ++pixel) {
        image.pixels[pixel * 3 + 0] = 255; // red
        image.pixels[pixel * 3 + 1] = 0;
        image.pixels[pixel * 3 + 2] = 128;
    }

    Preprocessing settings;
    const std::vector<float> tensor = cuvit::preprocess(image, settings);
    const std::size_t plane = static_cast<std::size_t>(settings.size) * settings.size;

    // The model reads [channels, height, width]; interleaved output would put
    // all three channels in the first plane.
    for (std::size_t index = 0; index < plane; ++index) {
        require(std::abs(tensor[index] - 1.0F) < 1.0e-5F, "the red plane is not saturated");
        require(std::abs(tensor[plane + index] + 1.0F) < 1.0e-5F, "the green plane is not empty");
        require(std::abs(tensor[2 * plane + index] - (128.0F / 255.0F - 0.5F) / 0.5F) < 1.0e-5F,
                "the blue plane has the wrong value");
    }
}

void test_centre_crop_keeps_the_middle() {
    // A wide image with a distinctive centre column: the crop must land on it
    // rather than on an edge.
    RgbImage image;
    image.width = 600;
    image.height = 200;
    image.pixels.assign(static_cast<std::size_t>(600) * 200 * 3, 0);
    for (int row = 0; row < 200; ++row) {
        for (int column = 250; column < 350; ++column) {
            const std::size_t base = (static_cast<std::size_t>(row) * 600 + column) * 3;
            image.pixels[base + 0] = 255;
            image.pixels[base + 1] = 255;
            image.pixels[base + 2] = 255;
        }
    }

    Preprocessing settings;
    const std::vector<float> tensor = cuvit::preprocess(image, settings);
    const std::size_t plane = static_cast<std::size_t>(settings.size) * settings.size;
    const std::size_t centre =
        static_cast<std::size_t>(settings.size / 2) * settings.size + settings.size / 2;

    require(tensor[centre] > 0.9F, "the centre of the crop is not the centre of the image");
    // The corners of the crop fall outside the white band.
    require(tensor[0] < -0.9F, "the crop reached past the white band");
    require(tensor[plane - 1] < -0.9F, "the crop reached past the white band");
}

void test_invalid_settings_are_rejected() {
    RgbImage image;
    image.width = 32;
    image.height = 32;
    image.pixels.assign(static_cast<std::size_t>(32) * 32 * 3, 100);

    Preprocessing settings;
    settings.crop_fraction = 0.0F;
    bool caught = false;
    try {
        static_cast<void>(cuvit::preprocess(image, settings));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a zero crop fraction was accepted");

    settings = Preprocessing{};
    settings.stddev = {1.0F, 0.0F, 1.0F};
    caught = false;
    try {
        static_cast<void>(cuvit::preprocess(image, settings));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "a zero standard deviation was accepted");

    caught = false;
    try {
        static_cast<void>(cuvit::preprocess(RgbImage{}));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "an empty image was accepted");
}

} // namespace

CUVIT_TEST_MAIN("image_test", {
    test_ppm_round_trip();
    test_malformed_images_are_rejected();
    test_preprocess_shape_and_normalization();
    test_preprocess_is_planar_and_keeps_channels_apart();
    test_centre_crop_keeps_the_middle();
    test_invalid_settings_are_rejected();
})
