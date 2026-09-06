# Cu-Vit

**A Vision Transformer (ViT) inference engine implemented from scratch in CUDA C++.**

Cu-Vit is an educational, performance-oriented implementation of the Vision
Transformer architecture ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)).
The project maps transformer operations directly to CUDA kernels instead of hiding
them behind a high-level deep learning framework.

> **Status:** Early development. The CUDA build, runtime error handling, GPU memory
> utilities, smoke executable, and initial test suite are operational. Model kernels
> and end-to-end inference are not implemented yet.

## Current capabilities

- CMake-based C++17/CUDA 17 build with configurable GPU architecture.
- Context-rich CUDA runtime error reporting through `CUVIT_CUDA_CHECK`.
- Move-only `DeviceBuffer<T>` with RAII ownership, checked host/device copies,
  device-to-device copies, bounds validation, and zero fill. Each buffer records
  the device it was allocated on, frees on that device, and rejects cross-device
  and self-overlapping copies.
- Move-only `HostBuffer<T>` holding page-locked host memory for staging.
- Move-only `Stream` owning a non-blocking CUDA stream.
- Non-owning strided `TensorView<T>` with slicing, axis selection, transposition,
  and reshaping, expressing the QKV split and per-head attention layouts without
  copying.
- `DeviceArena`, a 256-byte-aligned bump allocator placing all 152 ViT-Tiny
  tensors in one allocation.
- A versioned weight file format, its loader, and a timm exporter.
- Asynchronous `_async` counterparts for every transfer, so a full
  host-to-device, kernel, device-to-host path can run on one stream.
- Active CUDA device discovery and capability reporting, cached per device.
- A vector-add kernel used to verify compilation, launch, synchronization, and data
  transfer end to end.
- CTest coverage for memory ownership, data transfers, bounds checks, error
  propagation, page-locked allocation, stream lifetime, asynchronous pipelines,
  and kernel correctness across boundary sizes.
- Shared test support: seeded reproducible data, CPU reference implementations,
  and tolerance-aware comparisons that report which element disagreed.
- Clean `compute-sanitizer` memcheck runs for the executable and tests.

## Why Cu-Vit?

Most ViT implementations live behind PyTorch or TensorFlow abstractions. Cu-Vit
strips those away to answer a lower-level question: *what does it take to run a
Vision Transformer efficiently on a GPU?* The project is intended for:

- Learning how transformer math maps onto CUDA kernels and GPU memory hierarchy.
- Experimenting with kernel fusion, tiling, and precision strategies.
- Comparing hand-written kernels with cuBLAS and cuDNN baselines.

Correctness comes first: every kernel is checked against a CPU or library reference
before performance optimization begins.

## Target architecture

The first inference milestone targets ViT-Tiny/16 with a 224x224 input, batch size
one, and FP32 arithmetic. Training and backpropagation are outside the initial scope.

The planned inference path is:

```text
Image (C x H x W)
        |
        v
Patch embedding + CLS token + positional embedding
        |
        v
Transformer encoder blocks
  LayerNorm -> Multi-head attention -> residual
  LayerNorm -> MLP/GELU          -> residual
        |
        v
LayerNorm + classification head -> logits
```

## Requirements

- NVIDIA GPU with CUDA Compute Capability 7.0 or newer.
- CUDA Toolkit 11.x or 12.x. The selected architecture must be supported by the
  installed toolkit.
- CMake 3.18 or newer.
- A C++17-compatible host compiler.
- Ninja or Make.
- `compute-sanitizer` for GPU memory validation.
- cuBLAS and cuDNN will be optional reference backends in later stages.

Check the local toolchain with:

```bash
nvcc --version
nvidia-smi
cmake --version
```

## Build

The default build targets Ada GPUs (`sm_89`). Override
`CMAKE_CUDA_ARCHITECTURES` for another GPU generation.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build --parallel
```

Examples:

```bash
# Ampere
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86

# Volta
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70
```

Disable tests when only the library and executable are needed:

```bash
cmake -S . -B build -DCUVIT_BUILD_TESTS=OFF
```

## Run the CUDA smoke test

The current executable reports the active GPU and verifies a complete
host-to-device, kernel launch, synchronization, and device-to-host round trip:

```bash
./build/cu-vit
```

Example output:

```text
Cu-Vit
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Compute capability: 8.9
Global memory: 7.65 GiB
CUDA smoke test: PASS
```

## Tests

Run the full test suite with:

```bash
ctest --test-dir build --output-on-failure
```

Run GPU memory validation with:

```bash
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/device_buffer_test
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/host_buffer_test
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/stream_test
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/vector_add_test
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/cu-vit
```

## Transfers

`cudaMemcpyAsync` is only asynchronous when the host side is page-locked. A copy
issued from pageable memory is staged through a driver bounce buffer and blocks
the caller, so pair the `_async` transfers with `HostBuffer<T>`:

```cpp
cuvit::Stream stream;
cuvit::HostBuffer<float> host(count);
cuvit::DeviceBuffer<float> device(count);

device.copy_from_host_async(host.data(), host.size(), stream.get());
cuvit::launch_vector_add(a, b, out, count, stream.get());
device.copy_to_host_async(host.data(), host.size(), stream.get());
stream.synchronize();
```

The host buffer and the device buffer must both outlive the stream
synchronization.

## Testing approach

Every kernel is checked against a CPU reference in `tests/reference/`, written in
the plainest form the operation allows. Inputs come from a seeded generator, so a
failure can always be reproduced.

Choosing a tolerance matters more than it looks. An elementwise kernel rounds
identically to its reference and is held to bitwise equality. A reduction does
not: the GPU accumulates in a different order than the CPU, so results differ by
rounding alone. That error must be judged against the magnitude of the terms
being summed, not the magnitude of the result -- under cancellation a sum can be
far smaller than the values that produced it, making a negligible error look
enormous. Measured over dot products of normally distributed vectors, comparing a
32-wide blocked accumulation against a sequential one:

| Reduction length | Error vs. result | Error vs. summed terms |
| ---------------- | ---------------- | ---------------------- |
| 64               | 3.8e-05          | 2.0e-07                |
| 192              | 6.3e-05          | 2.3e-07                |
| 768              | 6.3e-04          | 1.9e-07                |
| 3072             | 3.2e-04          | 1.9e-07                |

The result-relative error moves by an order of magnitude and follows no usable
bound; the term-relative error holds near two ULP throughout. So `compare_close`
serves elementwise work, and `compare_close_scaled` takes the accumulated
magnitudes and serves reductions.

`numerics_test` checks these helpers against each other's failure modes, because
a comparison that silently accepts everything would make every other test pass
while proving nothing.

## Weights

Weights live in a flat versioned file: a header, a table of entries, then the
payloads, each aligned to 64 bytes. Reading it needs no third-party parser, and a
file written by an older exporter is refused by version rather than misread. A
tensor is fetched by name and checked against the shape the caller expects, since
a model that quietly ran with a mis-shaped weight would produce plausible output
and cost far more time than a failed load.

Export a timm checkpoint with:

```bash
python3 tools/export_vit_weights.py --model vit_tiny_patch16_224 --output vit_tiny.cvw
```

ViT-Tiny/16 yields 152 tensors totalling 5,717,416 parameters (21.8 MiB of
payload). The exported values are bit-identical to the PyTorch state dict.

## Memory layout

`DeviceBuffer` owns memory; `TensorView` only describes it. Keeping the two apart
means slicing and reshaping never raise a question about who frees what, and a
view stays cheap enough to pass by value.

Strides are explicit rather than implied by the shape, because attention reads
one buffer through several layouts: a packed QKV projection split into three
views, and `[tokens, heads, head_dim]` read as `[heads, tokens, head_dim]`. Both
are stride changes over memory that must not be copied. A view that is no longer
densely packed reports `is_contiguous() == false`, and `reshape` refuses it
rather than silently producing a wrong layout.

`DeviceArena` suballocates from a single `DeviceBuffer` with 256-byte alignment,
matching what `cudaMalloc` itself guarantees. ViT-Tiny's 152 weight tensors then
cost one allocation instead of 152.

## Project structure

```text
Cu-Vit/
├── include/cuvit/
│   ├── cuda_check.hpp       # CUDA error handling
│   ├── device_buffer.hpp    # Move-only GPU allocation owner
│   ├── device_info.hpp      # Active device metadata
│   ├── host_buffer.hpp      # Move-only page-locked host memory
│   ├── device_arena.hpp     # Aligned bump allocator over one allocation
│   ├── stream.hpp           # Move-only CUDA stream owner
│   ├── tensor.hpp           # Non-owning strided view
│   ├── vector_add.hpp       # Smoke-kernel interface
│   └── weights.hpp          # Weight file format and loader
├── src/
│   ├── kernels/             # CUDA kernels
│   ├── runtime/             # Runtime and device utilities
│   └── main.cpp             # Current smoke executable
├── tools/
│   └── export_vit_weights.py  # timm checkpoint -> .cvw
├── tests/
│   ├── reference/           # Naive CPU implementations kernels are judged against
│   ├── test_utils.hpp       # Comparisons, seeded data, test entry point
│   └── *.cu                 # CTest correctness tests
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Roadmap

- [x] Project skeleton and CMake CUDA build.
- [x] Device-memory utilities and CUDA error checking.
- [x] Streams, page-locked host staging, and asynchronous transfers.
- [x] Test support: seeded data, CPU references, tolerance-aware comparison.
- [x] Tensor shape, layout, and view abstractions.
- [x] Versioned weight format and pretrained ViT weight exporter.
- [ ] Tiled GEMM kernel with a cuBLAS reference.
- [ ] LayerNorm, GELU, and softmax kernels.
- [ ] Patch embedding, CLS token, and positional encoding.
- [ ] Multi-head self-attention.
- [ ] MLP and transformer encoder block.
- [ ] Full encoder stack and classification head.
- [ ] Image preprocessing and end-to-end inference.
- [ ] Benchmarks, kernel fusion, and FP16/mixed-precision optimization.

## References

- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## License

Cu-Vit is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
