#include "cuvit/image.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <ios>
#include <istream>
#include <stdexcept>
#include <string>

namespace cuvit {
namespace {

/// Reads the next PPM header field, skipping whitespace and # comments.
int read_header_value(std::istream& stream) {
    int value = -1;
    while (stream) {
        const int character = stream.peek();
        if (character == '#') {
            std::string comment;
            std::getline(stream, comment);
            continue;
        }
        if (std::isspace(character) != 0) {
            stream.get();
            continue;
        }
        if (std::isdigit(character) == 0) {
            break;
        }
        stream >> value;
        return value;
    }
    throw std::runtime_error("Malformed PPM header");
}

float sample_bilinear(const RgbImage& image, float x, float y, int channel) {
    const int left = static_cast<int>(std::floor(x));
    const int top = static_cast<int>(std::floor(y));
    const float dx = x - static_cast<float>(left);
    const float dy = y - static_cast<float>(top);

    const auto pixel = [&](int column, int row) {
        column = std::clamp(column, 0, image.width - 1);
        row = std::clamp(row, 0, image.height - 1);
        return static_cast<float>(
            image.pixels[(static_cast<std::size_t>(row) * image.width + column) * 3 + channel]);
    };

    const float top_edge = pixel(left, top) * (1.0F - dx) + pixel(left + 1, top) * dx;
    const float bottom_edge = pixel(left, top + 1) * (1.0F - dx) + pixel(left + 1, top + 1) * dx;
    return top_edge * (1.0F - dy) + bottom_edge * dy;
}

} // namespace

RgbImage load_ppm(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open image: " + path);
    }

    std::string magic;
    stream >> magic;
    if (magic != "P6") {
        throw std::runtime_error("Only binary PPM (P6) images are supported: " + path);
    }

    RgbImage image;
    image.width = read_header_value(stream);
    image.height = read_header_value(stream);
    const int maximum = read_header_value(stream);
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("PPM image has no extent: " + path);
    }
    if (maximum != 255) {
        throw std::runtime_error("Only 8-bit PPM images are supported: " + path);
    }
    stream.get(); // the single whitespace byte before the payload

    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height * 3);
    if (!stream.read(reinterpret_cast<char*>(image.pixels.data()),
                     static_cast<std::streamsize>(image.pixels.size()))) {
        throw std::runtime_error("PPM image ends before its payload does: " + path);
    }
    return image;
}

std::vector<float> preprocess(const RgbImage& image, const Preprocessing& settings) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty image");
    }
    if (settings.size <= 0) {
        throw std::invalid_argument("The preprocessing size must be positive");
    }
    if (!(settings.crop_fraction > 0.0F) || settings.crop_fraction > 1.0F) {
        throw std::invalid_argument("The crop fraction must lie in (0, 1]");
    }

    // Resize so the shorter side reaches size / crop_fraction, then take the
    // middle square. Both steps fold into one sampling pass: for each output
    // pixel, work back to the source coordinate it came from.
    const float resized = std::round(static_cast<float>(settings.size) / settings.crop_fraction);
    const int shorter = std::min(image.width, image.height);
    const float scale = static_cast<float>(shorter) / resized;

    const float crop = static_cast<float>(settings.size) * scale;
    const float origin_x = (static_cast<float>(image.width) - crop) * 0.5F;
    const float origin_y = (static_cast<float>(image.height) - crop) * 0.5F;
    const float step = crop / static_cast<float>(settings.size);

    const std::size_t plane = static_cast<std::size_t>(settings.size) * settings.size;
    std::vector<float> tensor(plane * 3);

    for (int channel = 0; channel < 3; ++channel) {
        const float mean = settings.mean[static_cast<std::size_t>(channel)];
        const float deviation = settings.stddev[static_cast<std::size_t>(channel)];
        if (deviation == 0.0F) {
            throw std::invalid_argument("A preprocessing standard deviation cannot be zero");
        }

        for (int row = 0; row < settings.size; ++row) {
            for (int column = 0; column < settings.size; ++column) {
                const float x = origin_x + (static_cast<float>(column) + 0.5F) * step - 0.5F;
                const float y = origin_y + (static_cast<float>(row) + 0.5F) * step - 0.5F;
                const float value = sample_bilinear(image, x, y, channel) / 255.0F;
                tensor[static_cast<std::size_t>(channel) * plane +
                       static_cast<std::size_t>(row) * settings.size + column] =
                    (value - mean) / deviation;
            }
        }
    }
    return tensor;
}

} // namespace cuvit
