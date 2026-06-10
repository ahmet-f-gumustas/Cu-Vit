# Cu-Vit

**A Vision Transformer (ViT) implemented from scratch in CUDA C++.**

Cu-Vit is an educational yet performance-oriented implementation of the Vision
Transformer architecture ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))
written directly against the CUDA runtime. The goal is to understand and
hand-optimize every component of a transformer — patch embedding, multi-head
self-attention, the MLP blocks, layer normalization and the classification head —
as raw GPU kernels, rather than relying on a high-level deep learning framework.

> **Status:** 🚧 Early development. The architecture and build system are being
> bootstrapped — see the [Roadmap](#roadmap) for what works and what's coming.

---

## Why Cu-Vit?

Most ViT code lives behind PyTorch or TensorFlow abstractions. Cu-Vit strips
those away to answer a simpler question: *what does it actually take to run a
Vision Transformer on a GPU?* It is meant for:

- Learning how transformer math maps onto CUDA kernels and memory hierarchy.
- Experimenting with custom kernel fusion, tiling and precision strategies.
- Benchmarking hand-written kernels against cuBLAS / cuDNN baselines.

---

## Architecture

The Vision Transformer processes an image as a sequence of patches:

```
Image (C×H×W)
   │
   ▼
┌──────────────────┐
│ Patch Embedding  │  split into P×P patches → linear projection → tokens
└──────────────────┘
   │  + [CLS] token
   │  + Positional Encoding
   ▼
┌──────────────────┐   ┌─ repeated ×L ──────────────────────────────┐
│ Transformer      │   │  LayerNorm → Multi-Head Self-Attention → +  │
│ Encoder Blocks   │ = │  LayerNorm → MLP (GELU) ───────────────→ +  │
└──────────────────┘   └─────────────────────────────────────────────┘
   │
   ▼
┌──────────────────┐
│ Classification   │  take [CLS] token → LayerNorm → Linear → logits
│ Head             │
└──────────────────┘
```

### Core components (CUDA kernels)

| Component                  | Description                                                        |
| -------------------------- | ------------------------------------------------------------------ |
| **Patch Embedding**        | Conv/linear projection of flattened image patches into tokens.     |
| **Positional Encoding**    | Learnable position embeddings added to the token sequence.         |
| **Multi-Head Attention**   | QKV projection, scaled dot-product attention, softmax, output proj. |
| **Layer Normalization**    | Per-token mean/variance normalization with affine parameters.      |
| **MLP / Feed-Forward**     | Two linear layers with a GELU activation in between.               |
| **Softmax**                | Numerically stable, row-wise softmax over attention scores.        |
| **GEMM**                   | Tiled matrix multiply (with optional cuBLAS fallback).             |

---

## Requirements

- **NVIDIA GPU** with CUDA Compute Capability 7.0+ (recommended)
- **CUDA Toolkit** 11.x or 12.x
- **CMake** ≥ 3.18
- A **C++17** compatible host compiler (GCC, Clang or MSVC)
- *(Optional)* cuBLAS / cuDNN for baseline comparisons

Check your toolkit:

```bash
nvcc --version
nvidia-smi
```

---

## Build

```bash
git clone https://github.com/ahmet-f-gumustas/Cu-Vit.git
cd Cu-Vit

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To target a specific GPU architecture (e.g. Ampere = `86`, Ada = `89`):

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

---

## Usage

> Usage is evolving alongside the implementation. Once the encoder is wired up,
> a minimal inference run will look roughly like:

```bash
./build/cu-vit --weights model.bin --image cat.png --patch-size 16
```

---

## Project structure

```
Cu-Vit/
├── src/            # CUDA kernels and model implementation
├── include/        # Public headers
├── tests/          # Unit / kernel correctness tests
├── benchmarks/     # Performance comparisons vs. cuBLAS/cuDNN
├── CMakeLists.txt  # Build configuration
├── LICENSE         # Apache License 2.0
└── README.md
```

*(Directories are added as the corresponding stages land — see the roadmap.)*

---

## Roadmap

- [ ] Project skeleton & CMake CUDA build
- [ ] Tensor / device-memory utilities and error checking
- [ ] Tiled GEMM kernel (+ cuBLAS reference)
- [ ] Patch embedding & positional encoding
- [ ] LayerNorm, GELU, softmax kernels
- [ ] Multi-head self-attention block
- [ ] Full transformer encoder stack
- [ ] Classification head & forward-pass inference
- [ ] Weight loading (import pretrained ViT weights)
- [ ] Kernel fusion & FP16 / mixed-precision optimization
- [ ] Benchmarks and correctness test suite

---

## References

- Dosovitskiy, A. et al. *An Image is Worth 16×16 Words: Transformers for Image
  Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE)
file for details.
