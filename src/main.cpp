#include "cuvit/device_info.hpp"
#include "cuvit/image.hpp"
#include "cuvit/stream.hpp"
#include "cuvit/vit.hpp"
#include "cuvit/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string weights;
    std::string image;
    std::string raw;
    int top = 5;
    int benchmark = 0;
};

void print_usage() {
    std::cout << "Usage: cu-vit --weights FILE.cvw (--image FILE.ppm | --raw FILE.bin)\n"
              << "              [--top N] [--benchmark ITERATIONS]\n\n"
              << "  --weights   Weight file from tools/export_vit_weights.py\n"
              << "  --image     Binary PPM (P6). Convert with:\n"
              << "                ffmpeg -i photo.jpg -pix_fmt rgb24 photo.ppm\n"
              << "  --raw       Preprocessed float32 tensor, 3x224x224, planar\n"
              << "  --top       Number of classes to report (default 5)\n"
              << "  --benchmark Time this many forward passes after a warm-up\n";
}

std::string require_value(int argc, char** argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string(flag) + " needs a value");
    }
    return argv[++index];
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--weights") {
            options.weights = require_value(argc, argv, index, "--weights");
        } else if (argument == "--image") {
            options.image = require_value(argc, argv, index, "--image");
        } else if (argument == "--raw") {
            options.raw = require_value(argc, argv, index, "--raw");
        } else if (argument == "--top") {
            options.top = std::stoi(require_value(argc, argv, index, "--top"));
        } else if (argument == "--benchmark") {
            options.benchmark = std::stoi(require_value(argc, argv, index, "--benchmark"));
        } else if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    return options;
}

std::vector<float> read_raw_tensor(const std::string& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open tensor: " + path);
    }
    std::vector<float> values(count);
    if (!stream.read(reinterpret_cast<char*>(values.data()),
                     static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("Tensor is shorter than the model's input: " + path);
    }
    return values;
}

void report_top(const std::vector<float>& logits, int count) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const int reported = std::min(count, static_cast<int>(logits.size()));
    std::partial_sort(order.begin(), order.begin() + reported, order.end(),
                      [&logits](int left, int right) { return logits[left] > logits[right]; });

    // Softmax over the logits, in the stable form, purely for reporting.
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (const float logit : logits) {
        total += std::exp(static_cast<double>(logit) - maximum);
    }

    std::cout << "\nTop " << reported << ":\n";
    for (int index = 0; index < reported; ++index) {
        const int label = order[static_cast<std::size_t>(index)];
        const double probability =
            std::exp(static_cast<double>(logits[static_cast<std::size_t>(label)]) - maximum) /
            total;
        std::cout << "  " << std::setw(4) << label << "  " << std::fixed << std::setprecision(4)
                  << std::setw(9) << logits[static_cast<std::size_t>(label)] << "  "
                  << std::setprecision(2) << std::setw(6) << probability * 100.0 << "%\n";
    }
}

void run_benchmark(cuvit::VisionTransformer& model, const std::vector<float>& image,
                   int iterations) {
    const cuvit::VitConfig& config = model.config();
    cuvit::DeviceBuffer<float> device_image(image.size());
    cuvit::DeviceBuffer<float> device_logits(static_cast<std::size_t>(config.classes));
    cuvit::Stream stream;
    device_image.copy_from_host(image.data(), image.size());

    for (int index = 0; index < 10; ++index) {
        model.forward(device_image.data(), device_logits.data(), stream.get());
    }
    stream.synchronize();

    const auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) {
        model.forward(device_image.data(), device_logits.data(), stream.get());
    }
    stream.synchronize();
    const auto finish = std::chrono::steady_clock::now();

    const double milliseconds =
        std::chrono::duration<double, std::milli>(finish - start).count() / iterations;
    std::cout << "\nForward pass: " << std::fixed << std::setprecision(3) << milliseconds
              << " ms  (" << std::setprecision(1) << 1000.0 / milliseconds << " images/s over "
              << iterations << " iterations)\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (options.weights.empty() || (options.image.empty() && options.raw.empty())) {
            print_usage();
            return 1;
        }

        const cuvit::DeviceInfo device = cuvit::current_device_info();
        std::cout << "Cu-Vit\n"
                  << "GPU: " << device.name << " (compute " << device.compute_major << '.'
                  << device.compute_minor << ")\n";

        const cuvit::WeightFile weights = cuvit::WeightFile::load(options.weights);
        const cuvit::VitConfig config;
        cuvit::VisionTransformer model(config, weights);
        std::cout << "Weights: " << weights.size() << " tensors, " << std::fixed
                  << std::setprecision(2)
                  << static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0)
                  << " MiB; activations "
                  << static_cast<double>(model.activation_bytes()) / (1024.0 * 1024.0) << " MiB\n";

        const std::size_t elements =
            static_cast<std::size_t>(config.channels) * config.image_size * config.image_size;
        std::vector<float> input;
        if (!options.raw.empty()) {
            input = read_raw_tensor(options.raw, elements);
            std::cout << "Input:   " << options.raw << " (preprocessed tensor)\n";
        } else {
            const cuvit::RgbImage image = cuvit::load_ppm(options.image);
            cuvit::Preprocessing settings;
            settings.size = config.image_size;
            input = cuvit::preprocess(image, settings);
            std::cout << "Input:   " << options.image << " (" << image.width << 'x' << image.height
                      << ")\n";
        }

        const std::vector<float> logits = model.forward(input);
        report_top(logits, options.top);

        if (options.benchmark > 0) {
            run_benchmark(model, input, options.benchmark);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cu-vit failed: " << error.what() << '\n';
        return 1;
    }
}
